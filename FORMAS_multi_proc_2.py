import sys
import os
import time
import numpy as np
sys.path.insert(0, os.path.join(os.getcwd(),'..'))
import CIBUSmod as cm

# Create session (Make sure that name and data_path match the notebook!)
session = cm.Session(
    name = 'FORMAS',
    data_path = '../CIBUSmod/data',
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

# Instantiate feed management
feed_mgmt = cm.FeedMgmt(
    herds = herds,
    par = cm.ParameterRetriever('FeedMgmt')
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
    crops = crops,
    herds = herds,
    par = cm.ParameterRetriever('CropResidueMgmt')
)

# Instantiate plant nutrient management
plant_nutrient_mgmt = cm.PlantNutrientMgmt(
    demand = demand,
    regions = regions,
    crops = crops,
    herds = herds,
    par = cm.ParameterRetriever('PlantNutrientMgmt')
)

# Instatiate machinery and energy management
machinery_and_energy_mgmt  = cm.MachineryAndEnergyMgmt(
    regions = regions,
    crops = crops,
    herds = herds,
    par = cm.ParameterRetriever('MachineryAndEnergyMgmt')
)

# Instatiate inputs management
inputs = cm.InputsMgmt(
    demand = demand,
    crops = crops,
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

def do_run(scn_year):
    scn, year = scn_year
    tic = time.time()

    years = session.scenarios()[scn]
    if year not in years:
        raise Exception('Dropped!')

    # Update all parameter values
    cm.ParameterRetriever.update_all_parameter_values(
        **session[scn],
        year = year
    )
    
    # Get region attributes
    regions.calculate()
    
    # Calculate food demand
    demand.calculate()
    
    # Calculate crops
    crops.calculate()
    
    # Calculate herds
    for h in herds:
        h.calculate()
    
    # Calculate feed
    feed_mgmt.calculate()

    if scn != 'baseline':
        # Get baseline crop areas, animal numbers and land use
        while True:
            try:
                baseline_ani = session.get_attr('geo','x_animals', scn='baseline').iloc[0]
                baseline_crp = session.get_attr('geo','x_crops', scn='baseline').iloc[0]
                baseline_lu = session.get_attr('c','area',{'region':None, 'crop':'land_use'}).iloc[0].unstack()
            except:
                time.sleep(10)
            else:
                break
    
    # Distribute animals and crops
    # Make optimisation problem
    if scn == 'baseline':
        geodist.make(
            use_cons=[1,2,3,4,5,6,7],
            scale_power=0.4
        )
        # Solve optimisation problem
        geodist.solve()
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
        C8_ani = baseline_ani.copy()

        if year == years[0]:
            sng_factors = [int(year)/100]
        else:
            sng_factors = np.linspace(
                int(year)/100,
                int(years[years.index(year)-1])/100,
                5,
                endpoint=False
            )

        drop_next = False
        for sng_factor in sng_factors:
            
            C9_crp = baseline_crp.loc[['Semi-natural pastures']] * sng_factor
            
            if 'incr' in scn:
                geodist.make(
                    use_cons=[1,2,3,4,5,6,7,8,9],
                    scale_power=0.4,
                    C8_crp = [ C8_SNG_P,   C8_SNG_PWT,   C8_SNG_M,   C8_FAL,   None,                     None                            ],
                    C8_ani = [ None,       None,         None,       None,      C8_ani.loc[['cattle']],   C8_ani.loc[['horses','sheep']]  ],
                    C8_rel = [ '>=',       '==',         '==',       '>=',     '>=',                     '=='                            ],
                    C9_crp = C9_crp
                )
                cm.helpers.drop_from_objective(geodist, 'crp', 'Ley for fodder')
                # cm.helpers.drop_from_objective(geodist, 'crp', 'Ley for grazing')
                # cm.helpers.drop_from_objective(geodist, 'crp', 'Maize for forage and silage')
                # cm.helpers.drop_from_objective(geodist, 'crp', 'Cereals for fodder')
                # cm.helpers.drop_from_objective(geodist, 'crp', 'Other crops for fodder')
                # cm.helpers.drop_from_objective(geodist, 'ani', 'cattle')
            elif 'free' in scn:
                geodist.make(
                    use_cons=[1,2,3,4,5,6,7,8,9],
                    scale_power=0.4,
                    C8_crp = [ C8_SNG_P,   C8_SNG_PWT,   C8_SNG_M,   C8_FAL,   None                            ],
                    C8_ani = [ None,       None,         None,       None,     C8_ani.loc[['horses','sheep']]  ],
                    C8_rel = [ '>=',       '==',         '==',       '>=',     '=='                            ],
                    C9_crp = C9_crp
                )
                geodist.x0['ani'].loc[['cattle']] = 0
                cm.helpers.drop_from_objective(geodist, 'crp', 'Ley for fodder')
                # cm.helpers.drop_from_objective(geodist, 'crp', 'Ley for grazing')
                # cm.helpers.drop_from_objective(geodist, 'crp', 'Maize for forage and silage')
                # cm.helpers.drop_from_objective(geodist, 'crp', 'Cereals for fodder')
                # cm.helpers.drop_from_objective(geodist, 'crp', 'Other crops for fodder')
                # cm.helpers.drop_from_objective(geodist, 'ani', 'cattle')
            else:
                geodist.make(
                    use_cons=[1,2,3,4,5,6,7,8,9],
                    scale_power=0.4,
                    C8_crp = [ C8_SNG_P,   C8_SNG_PWT,   C8_SNG_M,   C8_FAL,   None   ],
                    C8_ani = [ None,       None,         None,       None,     C8_ani ],
                    C8_rel = [ '>=',       '==',         '==',       '>=',     '=='   ],
                    C9_crp = C9_crp
                )
            geodist.x0['crp'].loc[['Ley for grazing']] = 0
            try:
                # Solve optimisation problem
                geodist.solve(
                    apply_solution = False
                )
            except:
                drop_next = True
                continue
            else:
                break

        if drop_next:
            # Remove comming "years" and raise exception
            years = session.scenarios()[scn]
            if year in years:
                session.update_scenario(
                    name = scn,
                    years = years[:years.index(year) + (1 if geodist.success else 0)],
                    prompt = False
                )

        if not geodist.success:
            raise Exception('No solution found')
        else:
            geodist.apply_solution()
    
    # Redistribute feeds (not yet implemented) and calculate enteric CH4 emissions
    feed_mgmt.calculate2()
    
    # Calculate manure
    manure_mgmt.calculate()
    
    # Calculate harvest of crop residues
    crop_residue_mgmt.calculate()
    
    # Calculate plant nutrient management
    plant_nutrient_mgmt.calculate()
    
    # Calculate energy requirements
    machinery_and_energy_mgmt.calculate()
    
    # Calculate inputs supply chain emissions
    inputs.calculate()
    
    # Store results (try again if first atempt fails)
    try:
        session.store(
            scn, year,
            demand, regions, crops, herds, geodist
        )
    except:
        session.store(
            scn, year,
            demand, regions, crops, herds, geodist
        )

    return time.time() - tic