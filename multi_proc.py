import sys
import os
import time
from contextlib import redirect_stdout, redirect_stderr
import pandas as pd
import numpy as np
import scipy
import cvxpy
import matplotlib.pyplot as plt
sys.path.insert(0, 'C:/Users/jnka0003/Git repos/CIBUSmod')
import CIBUSmod as cm
from CIBUSmod.utils.helpers import check_constraints

# Create session (Make sure that name and data_path match the notebook!)
session = cm.Session(
    name = 'FORMAS',
    data_path = 'C:\\Users/jnka0003/Git repos/CIBUSmod/data',
    data_path_scenarios = 'scenarios',
    data_path_output = 'output',
    timeout = 60 # Increase timeout to avoid failing to write if multiple processes try to write at the same time
)

# Instatiate Regions
regions = cm.Regions(
    par = cm.ParameterRetriever('Regions')
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

def _max_sng_obj(geodist):
    geodist.define_cvx_problem()

    # Get x variable
    x = geodist.problem.variables()[0]

    # Create objective
    rel = cm.ParameterRetriever.get_rel('crop','land_use')
    P = np.concatenate([
        np.zeros(len(geodist.x_idx_short['ani'])),
        np.array([1 if rel[cr] == 'semi-natural grasslands' else 0 for cr,_,_ in geodist.x_idx_short['crp']])
    ])
    obj = cvxpy.Maximize(
        cvxpy.sum(cvxpy.multiply(P, x))
    )
    
    # Create problem
    geodist.problem = cvxpy.Problem(
        objective = obj,
        constraints = geodist.problem.constraints
    )

def _make_ani_cons(geodist, name, M, b, rel):
    
    from CIBUSmod.optimisation.geo_dist import IndexedMatrix
    
    # Create A matrix
    M = scipy.sparse.csc_matrix(M)
    Z = scipy.sparse.csc_matrix((M.shape[0],len(geodist.x_idx_short['crp']))) # Zero matrix
    A = scipy.sparse.hstack([M,Z], format='csc')
    A = IndexedMatrix(
        matrix=A,
        row_idx=pd.Index([name]),
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

def _make_CH4_cons(geodist, feed_mgmt, baseline_CH4, factor):
    
    feed_mgmt.calculate2()
    
    # Get CH4 emissions per defining animal
    CH4 = _get_herds_par(herds, 'enteric_methane').sum(axis=1).reindex(geodist.x_idx_short['ani'])
    
    # Assert that indexes match
    assert (CH4.index == geodist.x_idx_short['ani']).all()
    
    _make_ani_cons(geodist, name='CH4', M=CH4, b=baseline_CH4 * factor, rel='<=')

    return None

def _make_milkmeat_cons(geodist, baseline_milkmeat):

    # Get milk and meat prod. per defining animal
    meat = _get_herds_par(herds, 'production').xs('meat', level='animal_prod', axis=1).sum(axis=1).xs('cattle', drop_level=False).reindex(geodist.x_idx_short['ani'], fill_value=0)
    milk = _get_herds_par(herds, 'production').xs('milk', level='animal_prod', axis=1).sum(axis=1).xs('cattle', drop_level=False).reindex(geodist.x_idx_short['ani'], fill_value=0)
    milkmeat = (milk - meat*baseline_milkmeat)
    _make_ani_cons(geodist, name='milk/meat', M=milkmeat, b=0, rel='==')

    return None

def _make_beeflamb_cons(geodist, baseline_beeflamb):

    # Get beef and lamb prod. per defining animal
    beef = _get_herds_par(herds, 'production').xs('meat', level='animal_prod', axis=1).sum(axis=1).xs('cattle', drop_level=False).reindex(geodist.x_idx_short['ani'], fill_value=0)
    lamb = _get_herds_par(herds, 'production').xs('meat', level='animal_prod', axis=1).sum(axis=1).xs('sheep', drop_level=False).reindex(geodist.x_idx_short['ani'], fill_value=0)
    beeflamb = (beef - lamb*baseline_beeflamb)
    _make_ani_cons(geodist, name='beef/lamb', M=beeflamb, b=0, rel='==')

    return None

def do_run(scn_year):
    scn, year = scn_year
    tic = time.time()

    with open(os.path.join(session.data_path_output, 'log', f'{scn}_{year}.log'), 'w') as f,\
        redirect_stdout(f), redirect_stderr(f):

        scn_str = f'Scenario: {scn}, {year}'
        print(f'\n{"-"*len(scn_str)}\n{scn_str}\n{"-"*len(scn_str)}\n')
        
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
    
        if scn != 'baseline':
            while True:
                try:
                    # Get baseline crop areas, animal numbers and land use
                    baseline_ani = session.get_attr('geo','x_animals', scn='baseline').iloc[0]
                    baseline_crp = session.get_attr('geo','x_crops', scn='baseline').iloc[0]
                    baseline_lu = session.get_attr('c','area',{'region':None, 'crop':'land_use'}, scn='baseline').iloc[0].unstack()
                    # Get baseline CH4 emissions
                    baseline_CH4 = session.get_attr(
                        'A', 'enteric_methane',
                        'none',
                        scn='baseline'
                    ).iloc[0]
                    # Get baseline milk/meat
                    prod = session.get_attr('A', 'prod', ['species', 'animal_prod'], scn='baseline').iloc[0]
                    baseline_milkmeat = prod[('cattle','milk')] / prod[('cattle','meat')]
                    # Get baseline beef/lamb
                    prod = session.get_attr('A', 'prod', ['species','animal_prod'], scn='baseline').iloc[0]
                    baseline_beeflamb = prod[('cattle','meat')] / prod[('sheep','meat')]
                except:
                    time.sleep(10)
                else:
                    break
        
        # Distribute animals and crops
        # Make optimisation problem
        if scn == 'baseline':
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
            .loc[['Fallow', 'Ley not harvested']]# * 0.8
            C8_ani = baseline_ani.copy()
        
            geodist.make(
                use_cons=[1,2,3,4,5,6,7,8],
                scale_power=0.4,
                C8_crp = [ C8_SNG_P,   C8_SNG_PWT,   C8_SNG_M,   C8_FAL,   None                    ],
                C8_ani = [ None,       None,         None,       None,     C8_ani.loc[['horses']]  ],
                C8_rel = [ '>=',       '==',         '==',       '>=',     '=='                    ],
                verbose=True
            )
            
            # Add constraint on CH4 emissions and milk/meat
            CH4_factor = float(year)/100
            _make_CH4_cons(geodist, feed_mgmt, baseline_CH4, CH4_factor)
            _make_milkmeat_cons(geodist, baseline_milkmeat)
            _make_beeflamb_cons(geodist, baseline_beeflamb)
        
            # First we solve with the obejctive of maximising semi-natural grassland area
            _max_sng_obj(geodist)
            geodist.solve(apply_solution=False, verbose=True)
        
            # Get semi-natural grassland areas from first solution and add constraint on total
            # semi-natural grassland area for second optimization round
            sng_areas = geodist.x['crp'].loc[['Semi-natural pastures', 'Semi-natural pastures, thin soils', 'Semi-natural pastures, wooded']]
            geodist.make_C9(C9_crp = sng_areas * 0.99, C9_rel = '>=') # Introduce a fair bit of slack to avoid unfeasible model
            geodist.make_C7()
        
            # Drop semi-natural grasslands from objective
            cm.helpers.drop_from_objective(geodist, 'crp', 'Semi-natural pastures')
            cm.helpers.drop_from_objective(geodist, 'crp', 'Semi-natural pastures, thin soils')
            cm.helpers.drop_from_objective(geodist, 'crp', 'Semi-natural pastures, wooded')
        
            # Solve optimisation problem again, this time minimising deviation from current
            # crop areas and animal numbers
            geodist.define_cvx_problem()
            geodist.solve(verbose=True)
        
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