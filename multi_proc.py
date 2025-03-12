import sys
import os
import time

from contextlib import redirect_stdout, redirect_stderr

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt

sys.path.insert(0, 'C:/Users/jnka0003/Git repos/CIBUSmod')
import CIBUSmod as cm
from CIBUSmod.utils.helpers import check_constraints

# Regions south of around 60 degrees north
south_of_60 = pd.Series({
    '111':True,
    '112':True,
    '311':True,
    '312':True,
    '321':True,
    '322':True,
    '411':True,
    '421':True,
    '422':True,
    '431':True,
    '511':True,
    '512':True,
    '513':True,
    '514':True,
    '515':True,
    '521':True,
    '611':True,
    '612':True,
    '621':True,
    '622':True,
    '711':True,
    '731':True,
    '811':True,
    '812':True,
    '813':True,
    '814':True,
    '821':True,
    '831':True,
    '911':True,
    '912':True,
    '913':True,
    '1011':True,
    '1111':True,
    '1112':True,
    '1121':True,
    '1122':True,
    '1123':True,
    '1124':True,
    '1131':True,
    '1211':True,
    '1212':True,
    '1213':True,
    '1214':True,
    '1215':True,
    '1216':True,
    '1221':True,
    '1222':True,
    '1311':True,
    '1321':True,
    '1322':True,
    '1331':True,
    '1411':True,
    '1412':True,
    '1421':True,
    '1511':True,
    '1512':True,
    '1521':True,
    '1522':True,
    '1611':True,
    '1612':True,
    '1613':True,
    '1614':True,
    '1615':True,
    '1616':True,
    '1617':True,
    '1621':True,
    '1622':True,
    '1623':True,
    '1711':True,
    '1712':True,
    '1713':True,
    '1721':True,
    '1722':True,
    '1723':True,
    '1724':False,
    '1811':True,
    '1812':True,
    '1813':True,
    '1821':True,
    '1911':True,
    '1912':True,
    '1921':True,
    '1922':True,
    '2011':False,
    '2012':False,
    '2019':False,
    '2111':False,
    '2121':False,
    '2122':False,
    '2211':False,
    '2212':False,
    '2221':False,
    '2311':False,
    '2312':False,
    '2319':False,
    '2331':False,
    '2411':False,
    '2412':False,
    '2413':False,
    '2414':False,
    '2415':False,
    '2419':False,
    '2511':False,
    '2512':False,
    '2519':False,
    '2521':False
}).rename_axis('region')

def _max_sng_obj_alt1(crops, geodist):

    import cvxpy

    print('Making max SNG objective (alt. 1) ...')

    yields = crops.data_attr.get('harvest')

    geodist.define_cvx_problem()

    # Get x variable
    x = geodist.problem.variables()[0]

    # Create objective
    rel = cm.ParameterRetriever.get_rel('crop','land_use')
    P = np.concatenate([
        np.zeros(len(geodist.x_idx_short['ani'])),
        np.array([yields.loc[(cr,ps,re)] if rel[cr] == 'semi-natural grasslands' else 0 for cr,ps,re in geodist.x_idx_short['crp']])
    ])
    obj = cvxpy.Maximize(
        cvxpy.sum(cvxpy.multiply(P, x))
    )
    
    # Create problem
    geodist.problem = cvxpy.Problem(
        objective = obj,
        constraints = geodist.problem.constraints
    )

    return None

def _max_sng_obj_alt2(geodist):

    print('Making max SNG objective (alt. 1) ...')

    for w in ['ani','crp']:
        for k in geodist.x0_idx[w].unique(0):
            if k not in ['Semi-natural pastures', 'Semi-natural pastures, wooded', 'Semi-natural pastures, thin soils', 'Semi-natural meadows']:
                cm.helpers.drop_from_objective(geodist, which=w, key=k)

    return None

def _make_ani_cons(geodist, name, M, b, rel):
    
    from CIBUSmod.optimisation.geo_dist import IndexedMatrix

    if isinstance(M, pd.DataFrame):
        row_idx = M.index
    else:
        row_idx = pd.Index([name])

    # Create A matrix
    M = scipy.sparse.csc_matrix(M)
    Z = scipy.sparse.csc_matrix((M.shape[0],len(geodist.x_idx_short['crp']))) # Zero matrix
    A = scipy.sparse.hstack([M,Z], format='csc')
    A = IndexedMatrix(
        matrix=A,
        row_idx=row_idx,
        col_idx={k:v.copy() for k,v in geodist.x_idx_short.items()}
    )

    # Append constraint
    geodist.constraints.update({f'{name}: A @ x {rel} b' : {
        'left' : lambda x,A: A.M @ x,
        'right' : lambda A: b,
        'rel' : rel,
        'pars' : {'A':A}
    }})

    return None

def _get_herds_par(herds, attr):
    
    res = pd.concat([
        pd.concat({h.species: 
            pd.concat({h.breed:
                pd.concat({h.prod_system:
                    pd.concat({h.sub_system:
                        h.data_attr.get(attr)
                    }, names=['sub_system'])
                }, names=['prod_system'])
            }, names=['breed'])
        }, names=['species'])
    for h in herds])

    return res

def _make_CH4_cons(herds, geodist, feed_mgmt, baseline_CH4, factor):

    print('Making CH4 constraint ...')
    
    feed_mgmt.calculate2()
    
    # Get CH4 emissions per defining animal
    CH4 = _get_herds_par(herds, 'enteric_methane').sum(axis=1).reindex(geodist.x_idx_short['ani'])
    
    # Assert that indexes match
    assert (CH4.index == geodist.x_idx_short['ani']).all()
    
    _make_ani_cons(geodist, name='CH4', M=CH4, b=baseline_CH4 * factor, rel='<=')

    return None

def _make_milkmeat_cons(herds, geodist, baseline_milkmeat):

    print('Making milk/meat constraint ...')

    # Get milk and meat prod. per defining animal
    meat = _get_herds_par(herds, 'production').xs('meat', level='animal_prod', axis=1).sum(axis=1).xs('cattle', drop_level=False).reindex(geodist.x_idx_short['ani'], fill_value=0)
    milk = _get_herds_par(herds, 'production').xs('milk', level='animal_prod', axis=1).sum(axis=1).xs('cattle', drop_level=False).reindex(geodist.x_idx_short['ani'], fill_value=0)
    milkmeat = (milk - meat*baseline_milkmeat)
    _make_ani_cons(geodist, name='milk/meat', M=milkmeat, b=0, rel='==')

    return None

def _make_orgcon_cons(geodist, baseline_org_per_con):

    print('Making org/con constraint ...')
    
    d = [[(-1/baseline_org_per_con.loc[sp,br] if ps=='organic' else 1) if (sp==sp2) and (br==br2) else 0 for sp,br,ps,_,_ in geodist.x_idx_short['ani']] for sp2,br2 in baseline_org_per_con.index]
    M = pd.DataFrame(
        d,
        index = baseline_org_per_con.index,
        columns = geodist.x_idx_short['ani']
    )
    _make_ani_cons(geodist, name='org/con', M=M, b=0, rel='==')
    
    return None

def _make_beeflamb_cons(herds, geodist, baseline_beeflamb):

    print('Making beef/lamb constraint ...')

    # Get beef and lamb prod. per defining animal
    beef = _get_herds_par(herds, 'production').xs('meat', level='animal_prod', axis=1).sum(axis=1).xs('cattle', drop_level=False).reindex(geodist.x_idx_short['ani'], fill_value=0)
    lamb = _get_herds_par(herds, 'production').xs('meat', level='animal_prod', axis=1).sum(axis=1).xs('sheep', drop_level=False).reindex(geodist.x_idx_short['ani'], fill_value=0)
    beeflamb = (beef - lamb*baseline_beeflamb)
    _make_ani_cons(geodist, name='beef/lamb', M=beeflamb, b=0, rel='==')

    return None

def _make_sng_cons(geodist, sng_areas, tol=0.001):

    print('Making SNG constraint ...')

    geodist.make_C9(C9_crp = sng_areas * (1-tol), C9_rel = '>=')
    geodist.make_C7()

def do_run(session, scn_year):

    # Activate session in environment
    session.activate()

    scn, year = scn_year

    # Create log folder if it does not exist
    log_path = os.path.join(session.data_path_output, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    with open(os.path.join(log_path, f'{scn}_{year}.log'), 'w') as f,\
        redirect_stdout(f), redirect_stderr(f):

        # Print path and time-stamp
        print(
            session.data_path,
            time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            '',
            sep='\n'
        )


        tic = time.time()

        # Increase timeout to avoid failing to write if multiple processes try to write at the same time
        session.db_timeout = 60

        ###############################
        ###   INSTANTIATE MODULES   ###
        ###############################

        print('CREATING MODULES')

        # Instatiate Regions
        regions = cm.Regions(
            par = cm.ParameterRetriever('Regions'),
            settings = {'max_land_use_from_scenario_x0' : True}
        )

        # Instantiate DemandAndConversions
        demand = cm.DemandAndConversions(
            par = cm.ParameterRetriever('DemandAndConversions')
        )

        # Instantiate CropProduction
        crops = cm.CropProduction(
            par = cm.ParameterRetriever('CropProduction'),
            index = regions.data_attr.get('x0_crops').index
        )    

        # Instantiate AnimalHerds
        # Each AnimalHerd object is stored in an indexed pandas.Series
        herds = cm.make_herds(regions)

        # If Nature conservation horses scenario add new horse breed
        if session[scn]['scenario_workbooks'] is not None and 'NAT_HORSES' in session[scn]['scenario_workbooks']:
            nat_horses = True
            h = cm.HorseHerd(
                par = cm.ParameterRetriever('HorseHerd'),
                index = regions.data_attr.get('x0_animals').index.get_level_values('region').unique(),
                breed = 'conservation horses',
                prod_system = 'conventional',
                sub_system = 'none'
            )
            s = pd.Series(
                [h],
                index = pd.MultiIndex.from_tuples(
                    [(h.species, h.breed, h.prod_system, h.sub_system)],
                    names = ['species','breed','prod_system','sub_system']
                )
            )
            herds = pd.concat([herds, s])
        else:
            nat_horses = False

        # Instantiate WasteAndCircularity
        waste = cm.WasteAndCircularity(
            demand = demand,
            crops = crops,
            herds = herds,
            par = cm.ParameterRetriever('WasteAndCircularity')
        )

        # Instantiate WasteAndCircularity
        waste = cm.WasteAndCircularity(
            demand = demand,
            crops = crops,
            herds = herds,
            par = cm.ParameterRetriever('WasteAndCircularity')
        )

        # Instantiate feed management
        feed_mgmt = cm.FeedMgmt(
            herds = herds,
            par = cm.ParameterRetriever('FeedMgmt')
        )

        # Instantiate by-product management
        byprod_mgmt = cm.ByProductMgmt(
            demand = demand,
            herds = herds,
            par = cm.ParameterRetriever('ByProductMgmt')
        )

        # Instantiate manure management
        manure_mgmt = cm.ManureMgmt(
            herds = herds,
            feed_mgmt = feed_mgmt,
            par = cm.ParameterRetriever('ManureMgmt'),
            settings = {
                'NPK_excretion_from_balance' : True
            }
        )

        # Instantiate crop residue managment
        crop_residue_mgmt = cm.CropResidueMgmt(
            demand = demand,
            crops = crops,
            herds = herds,
            par = cm.ParameterRetriever('CropResidueMgmt')
        )

        # Instantiate plant nutrient management
        plant_nutrient_mgmt = cm.PlantNutrientMgmt(
            demand = demand,
            regions = regions,
            crops = crops,
            waste = waste,
            herds = herds,
            par = cm.ParameterRetriever('PlantNutrientMgmt')
        )

        # Instatiate machinery and energy management
        machinery_and_energy_mgmt  = cm.MachineryAndEnergyMgmt(
            regions = regions,
            crops = crops,
            waste = waste,
            herds = herds,
            par = cm.ParameterRetriever('MachineryAndEnergyMgmt')
        )

        # Instatiate inputs management
        inputs = cm.InputsMgmt(
            demand = demand,
            crops = crops,
            waste = waste,
            herds = herds,
            par = cm.ParameterRetriever('InputsMgmt')
        )

        # Instantiate geo distributor
        geodist = cm.GeoDistributor(
            regions = regions,
            demand = demand,
            crops = crops,
            herds = herds,
            feed_mgmt = feed_mgmt,
            par = cm.ParameterRetriever('GeoDistributor')
        )

        ############################
        ###   RUN CALCULATIONS   ###
        ############################

        print('STARTING MODEL RUN')

        # Update all parameter values
        cm.ParameterRetriever.update_all_parameter_values(
            **session[scn],
            year = year
        )

        # Get region attributes
        regions.calculate(verbose=True)

        # Calculate food demand
        demand.calculate(verbose=True)

        # Calculate crops
        crops.calculate(verbose=True)

        # Calculate herds
        for h in herds:
            h.calculate(verbose=True)

        # Calculate feed
        feed_mgmt.calculate(verbose=True)

        # Get data from baseline
        if scn != 'BL':
            while True:
                try:
                    # Get baseline crop areas, animal numbers and land use
                    baseline_ani = session.get_attr('geo','x_animals', scn='BL').iloc[0]
                    baseline_crp = session.get_attr('geo','x_crops', scn='BL').iloc[0]
                    baseline_lu = session.get_attr('c','area',{'region':None, 'crop':'land_use'}, scn='BL').iloc[0].unstack()
                    # Get baseline CH4 emissions
                    baseline_CH4 = session.get_attr(
                        'A', 'enteric_methane',
                        'none',
                        scn='BL'
                    ).iloc[0]
                    # Get baseline milk/meat
                    prod = session.get_attr('A', 'prod', ['species', 'animal_prod'], scn='BL').iloc[0]
                    baseline_milkmeat = prod[('cattle','milk')] / prod[('cattle','meat')]
                    # Get baseline beef/lamb
                    prod = session.get_attr('A', 'prod', ['species','animal_prod'], scn='BL').iloc[0]
                    baseline_beeflamb = prod[('cattle','meat')] / prod[('sheep','meat')]
                    # Get baseline org/con
                    heads = session.get_attr('G','x_ani',['species','breed','prod_system'], scn='BL').iloc[0].loc[['cattle','sheep']]
                    baseline_org_per_con = (heads.xs('organic', level='prod_system') / heads.xs('conventional', level='prod_system'))
                except:
                    time.sleep(10)
                else:
                    break

        # Distribute animals and crops
        # Make optimisation problem
        if scn == 'BL':
            geodist.make(
                use_cons=[1,2,3,4,5,6,7],
                scale_power=0.4,
                verbose=True
            )
            # Solve optimisation problem
            geodist.solve(verbose=True)
        else:
            # Drop demand for cattle, sheep and horses
            demand.data_attr.update(
                'animal_prod_demand',
                demand.data_attr.get('animal_prod_demand')
                .loc[(slice(None),['pigs','poultry'],slice(None))]
            )

            # Set maximum cropland and greenhouse area to baseline levels
            regions.data_attr.get('max_land_use').update(baseline_lu.loc[:,['cropland','greenhouse']])

            # Baseline Semi-natural grassland areas
            C8_SNG_P = baseline_crp.copy()\
            .loc[['Semi-natural pastures']]
            C8_SNG_PWT = baseline_crp.copy()\
            .loc[['Semi-natural pastures, wooded','Semi-natural pastures, thin soils']]
            C8_SNG_M = baseline_crp.copy()\
            .loc[['Semi-natural meadows']]
            C8_FAL = baseline_crp.copy()\
            .loc[['Fallow', 'Ley not harvested']]
            C8_FOD = baseline_crp.copy()\
            .loc[['Cereals for fodder', 'Other crops for fodder']]
            if 'fix ani' in scn:
                # All animals fixed
                C8_ANI = baseline_ani.copy()
            else:
                # Pigs, pultry and horses fixed
                C8_ANI = baseline_ani.copy().loc[['horses','pigs','poultry']]

            for opt_nr in [1,2]:
                print(f'Optimisation round #{opt_nr}')
                geodist.make(
                    use_cons=[1,2,3,4,5,6,7,8],
                    scale_power=0 if opt_nr == 1 else 0.4,
                    C8_crp = [ C8_SNG_P,   C8_SNG_PWT,   C8_SNG_M,   C8_FAL,  C8_FOD,   None    ],
                    C8_ani = [ None,       None,         None,       None,    None,     C8_ANI*1.001  ],
                    C8_rel = [ '>=',       '==',         '==',       '>=',    '<=',     '<='    ],
                    verbose=True
                )

                if nat_horses:
                    # No nature conservation hoses below ~60 degrees north
                    idx = pd.IndexSlice
                    nat_horses_lim = pd.Series(
                        0.0,
                        index = geodist.x_idx['ani'].to_frame()
                        .loc[idx[
                            ['horses'],
                            ['conservation horses'],
                            :,
                            :,
                            south_of_60[~south_of_60].index
                        ]].index
                    )
                    geodist.make_C8(
                        C8_ani = nat_horses_lim,
                        C8_rel = '<='
                    )
                    geodist.make_C7()
                    # Drop conservation horses from objective
                    cm.helpers.drop_from_objective(geodist, which='ani', key=('horses','conservation horses'))
                    
                
                # Add constraint on CH4 emissions and milk/meat
                if not 'fix ani' in scn:
                    CH4_factor = float(year)/100
                    _make_CH4_cons(herds, geodist, feed_mgmt, baseline_CH4, CH4_factor)
                if not 'fix ani' in scn:
                    _make_milkmeat_cons(herds, geodist, baseline_milkmeat)
                    _make_beeflamb_cons(herds, geodist, baseline_beeflamb)
                    _make_orgcon_cons(geodist, baseline_org_per_con)

                if opt_nr == 1:
                    # First we solve while dropping everything from the
                    # obejctive except for semi-natural grasslands
                    _max_sng_obj_alt1(crops, geodist)
                    geodist.solve(apply_solution=False, verbose=True)
                    
                    # Get semi-natural grassland areas from first solution to constrain
                    # semi-natural grassland area per region for second optimization round
                    sng_areas = geodist.x['crp'].loc[['Semi-natural pastures', 'Semi-natural pastures, thin soils', 'Semi-natural pastures, wooded']]
                    print(f'SNG area: {sng_areas.sum()/1_000_000:.2f} Mha')
                elif opt_nr == 2:           
                    # Solve optimisation problem again, this time minimising deviation from current
                    # crop areas and animal numbers
                    _make_sng_cons(geodist, sng_areas, tol=0.001)
                    for tol in [1e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5]:
                        try:
                            print(f'BarConvTol = {tol:.1e}')
                            geodist.solve(solver_settings={'solver':'GUROBI', 'BarConvTol':tol}, verbose=True)
                        except:
                            print('')
                            continue
                        else:
                            break
                    sng_areas = geodist.x['crp'].loc[['Semi-natural pastures', 'Semi-natural pastures, thin soils', 'Semi-natural pastures, wooded']]
                    print(f'SNG area: {sng_areas.sum()/1_000_000:.2f} Mha')

        # Redistribute feeds (not yet implemented) and calculate enteric CH4 emissions
        feed_mgmt.calculate2(verbose=True)

        # Balance by-product demand and suply
        byprod_mgmt.calculate(verbose=True)

        # Calculate manure
        manure_mgmt.calculate(verbose=True)

        # Calculate harvest of crop residues
        crop_residue_mgmt.calculate(verbose=True)

        # Calculate treatment of wastes and other feedstocks
        waste.calculate(verbose=True)

        # Calculate plant nutrient management
        plant_nutrient_mgmt.calculate(verbose=True)

        # Calculate energy requirements
        machinery_and_energy_mgmt.calculate(verbose=True)

        # Calculate inputs supply chain emissions
        inputs.calculate(verbose=True)

        # Store results (try again if first atempt fails)
        try:
            session.store(
                scn, year,
                demand, regions, crops, herds, waste, geodist
            )
        except:
            session.store(
                scn, year,
                demand, regions, crops, herds, waste, geodist
            )

        t = time.time() - tic
        m = int(t/60)
        s = int(round(t - m*60))
        print(f'{scn}, {year} finished successfully in {m}min {s}s')

        check_constraints(geodist)
        plt.savefig(os.path.join(session.data_path_output, 'log', f'{scn}_{year}_cons.png'))

    return t