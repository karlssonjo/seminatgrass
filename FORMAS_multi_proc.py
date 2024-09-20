import sys
import os
import time
sys.path.insert(0, os.path.join(os.getcwd(),'..'))
import CIBUSmod as cm

# Create session (Make sure that name and data_path match the notebook!)
session = cm.Session(
    name = 'FORMAS',
    data_path = '../CIBUSmod/data',
    timeout = 60 # Increase timeout to avoid failing to write if multiple processes try to write at the same time
)
cm.ParameterRetriever.data_path_scenarios = os.path.join('scenarios')

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

    # Baseline animal numbers
    if scn != 'baseline':
        while True:
            try:
                C8_ani = session.get_attr('geo','x_animals', scn='baseline').iloc[0]\
                .loc[['cattle','sheep','horses']]
                baseline_crop_areas = session.get_attr('geo','x_crops', scn='baseline').iloc[0]
                C8_SNG_P = baseline_crop_areas.copy()\
                .loc[['Semi-natural pastures']]
                C8_SNG_PWT = baseline_crop_areas.copy()\
                .loc[['Semi-natural pastures, wooded','Semi-natural pastures, thin soils']]
                C8_SNG_M = baseline_crop_areas.copy()\
                .loc[['Semi-natural meadows']]
            except:
                time.sleep(10)
            else:
                break

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

    # Aim for more semi-natural pastures
    if scn != 'baseline':
        regions.data_attr.get('x0_crops').loc[['Semi-natural pastures']] *= 1e12
    
    # Distribute animals and crops
    # Make optimisation problem
    if scn == 'baseline':
        geodist.make(
            use_cons=[1,2,3,4,5,6,7],
            scale_power=0.4
        )
    else:
        demand.data_attr.update(
            'animal_prod_demand',
            demand.data_attr.get('animal_prod_demand')
            .loc[(slice(None),['pigs','poultry'],slice(None))]
        )
        if scn in ['max_cur','steers']:
            geodist.make(
                use_cons=[1,2,3,4,5,6,7,8],
                scale_power=0.4,
                C8_crp = [ C8_SNG_P,   C8_SNG_PWT,   C8_SNG_M,   None   ],
                C8_ani = [ None,       None,         None,       C8_ani ],
                C8_rel = [ '>=',       '==',         '==',       '=='   ]
            )
        else:
            geodist.make(
                use_cons=[1,2,3,4,5,6,7,8],
                scale_power=0.4,
                C8_crp = [ C8_SNG_P,   C8_SNG_PWT,   C8_SNG_M,   None,                             None                            ],
                C8_ani = [ None,       None,         None,       C8_ani.loc[['horses','sheep']],   C8_ani.loc[(['cattle'],['dairy'])]  ],
                C8_rel = [ '>=',       '==',         '==',       '==',                             '=='                            ]
            )
            # Drop cattle and fodder crops from objective function
            cm.helpers.drop_from_objective(geodist, 'ani', 'cattle')
            cm.helpers.drop_from_objective(geodist, 'crp', 'Ley for fodder')
            cm.helpers.drop_from_objective(geodist, 'crp', 'Ley for grazing')
            cm.helpers.drop_from_objective(geodist, 'crp', 'Maize for forage and silage')
            cm.helpers.drop_from_objective(geodist, 'crp', 'Cereals for fodder')
            cm.helpers.drop_from_objective(geodist, 'crp', 'Other crops for fodder')

    # Solve optimisation problem
    geodist.solve(
        solver_settings = {
            'solver':'OSQP',
            'max_iter':200000,
            'eps_abs':5e-6,
            'eps_rel':5e-6,
            'verbose':False
        }
    )
    
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