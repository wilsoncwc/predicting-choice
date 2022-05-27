# Directories to store generated data and metrics
project_root = '/homes/wwc4618/predicting-choice'
dataset_root = '/vol/bitbucket/wwc4618/datasets'

# SSx dataset constants
full_dataset_label = 'No Bounds'
training_places = ['London']

# OSMnx constants
osmnx_buffer = 10000

# Fields to ignore
meridian_fields = ['meridian_id', 'meridian_gid', 'meridian_code',
                   'meridian_osodr', 'meridian_number', 'meridian_road_name',
                   'meridian_indicator', 'meridian_class', 'meridian_class_scale']
census_geom_fields = ['wz11cd', 'lsoa11nm', 'msoa11nm',
                      'oa11cd', 'lsoa11cd', 'msoa11cd'] # Allowed: lad11cd, lad11nm
fields_to_ignore = meridian_fields + census_geom_fields + ['id', 'accident_count']
ignore_non_accident_field = meridian_fields + census_geom_fields + ['id']

# Included fields
unnorm_feature_fields = ['metres', 'choice2km', 'nodecount2km', 'integration2km',
                      'choice10km', 'nodecount10km','integration10km',
                      'choice100km','nodecount100km','integration100km']
rank_fields = ['choice2kmrank', 'choice10kmrank','integration10kmrank', 'integration2kmrank']
log_fields = ['choice2kmlog','choice10kmlog','choice100kmlog']
all_feature_fields = unnorm_feature_fields + rank_fields + log_fields

# Field Combinations
km2_fields = ['choice2km', 'nodecount2km', 'integration2km']
km10_fields = ['choice10km', 'nodecount10km', 'integration10km']
km100_fields = ['choice100km', 'nodecount100km', 'integration100km']
choice_fields = ['choice2km', 'choice10km', 'choice100km']
integration_fields = ['integration2km', 'integration10km', 'integration100km']
nodecount_fields = ['nodecount2km', 'nodecount10km', 'nodecount100km']

# Number of point geometries for line features (dual graph)
NUM_GEOM = 5

# Post-processing features
primal_feats = ['metres', 'choice2km', 'nodecount2km', 'integration2km', 'choice10km',
         'nodecount10km', 'integration10km', 'choice100km', 'nodecount100km',
         'integration100km', 'choice2kmrank', 'choice10kmrank', 'integration10kmrank',
         'integration2kmrank', 'choice2kmlog', 'choice10kmlog', 'choice100kmlog', 'x', 'y']

geom_feats = ['metres', 'mid_x', 'mid_y'] + \
    [f'geom{i}_{ax}' for i in range(NUM_GEOM) for ax in ('x', 'y')]
dual_feats = geom_feats + \
    ['choice2km', 'nodecount2km', 'integration2km', 'choice10km', 'nodecount10km',
     'integration10km', 'choice100km', 'nodecount100km', 'integration100km',
     'choice2kmrank', 'choice10kmrank', 'integration10kmrank', 'integration2kmrank',
     'choice2kmlog', 'choice10kmlog', 'choice100kmlog']

# All SSx local authorities
included_places = ['Isle of Wight','Wycombe','Enfield','Slough','South Bucks','Hillingdon',
    'Ealing','Chiltern','Copeland','Windsor and Maidenhead','Plymouth',
    'South Hams','Oxford','Waltham Forest','Mendip','Dudley','Cotswold',
    'Erewash','Redbridge','Epping Forest','Test Valley',
    'Basingstoke and Deane','South Gloucestershire','Woking','Broxbourne',
    'Wolverhampton','Wiltshire','Swindon','Bath and North East Somerset',
    'Trafford','Salford','South Staffordshire','West Oxfordshire',
    'Malvern Hills','Vale of White Horse','South Kesteven','North Kesteven',
    'Guildford','Southwark','Chichester','Waverley','Elmbridge',
    'Forest of Dean','Tewkesbury','Charnwood','Sheffield','Ashfield',
    'North West Leicestershire','North East Derbyshire','Stroud','Shropshire',
    'Telford and Wrekin','Horsham','City of London','Newcastle-under-Lyme',
    'Stafford','Stoke-on-Trent','Arun','Lichfield','Sandwell','Birmingham',
    'Amber Valley','Mole Valley','Bolsover','Rushcliffe','Wychavon','Gedling',
    'Gloucester','Liverpool','Newark and Sherwood','Sefton','St. Helens',
    'Worcester','Flintshire','Bassetlaw','Knowsley','Wyre Forest',
    'Bromsgrove','Denbighshire','Herefordshire, County of','Rotherham',
    'Chesterfield','Barnet','Monmouthshire','Cheltenham','Spelthorne',
    'Sutton','Cheshire East','Cheshire West and Chester','Runnymede',
    'Reigate and Banstead','Kingston upon Thames','Epsom and Ewell',
    'Tandridge','Merton','Croydon','Stockport','Stratford-on-Avon',
    'Northumberland','Walsall','Solihull','Mid Sussex','Waveney','Lincoln',
    'Tonbridge and Malling','Redditch','Doncaster','Brighton and Hove',
    'Tameside','High Peak','North Lincolnshire','Lewes','North Warwickshire',
    'Gravesham','East Riding of Yorkshire','South Somerset','West Dorset',
    'Sevenoaks','Medway','Nottingham','Tamworth','Warwick','Dartford',
    'Surrey Heath','South Norfolk','Oldham','West Lindsey','Lambeth',
    'North Norfolk','Breckland','Broxtowe','Staffordshire Moorlands',
    'Worthing',"King's Lynn and West Norfolk",'Kirklees','Calderdale',
    'Great Yarmouth','Crawley','East Devon','Mid Devon','St Edmundsbury',
    'Mid Suffolk','East Cambridgeshire','Forest Heath','North Devon',
    'Fenland','North Dorset','Bracknell Forest','Haringey','Coventry',
    'Hounslow','Wakefield','Eastbourne','Selby','Wealden','West Somerset',
    'Exeter','Kingston upon Hull, City of','Broadland','Norwich','Eden',
    'Rugby','Barrow-in-Furness','Cannock Chase','South Lakeland','Carlisle',
    'Isle of Anglesey','West Berkshire','Allerdale','Derby',
    'South Derbyshire','Leicester','Hinckley and Bosworth','Gwynedd',
    'County Durham','Bexley','West Devon','Torridge','Winchester',
    'South Oxfordshire','Blaby','North Somerset','Bristol, City of','Wigan',
    'Warrington','West Lancashire','Fylde','Manchester','North Tyneside',
    'Newcastle upon Tyne','Rochdale','Barnsley','Gateshead','Sunderland',
    'Redcar and Cleveland','Hartlepool','Bury','Ryedale','Harborough',
    'Rutland','Scarborough','Harrogate','Melton','Hambleton','Richmondshire',
    'South Tyneside','Pendle','Bolton','Bradford','Chorley','York','Leeds',
    'New Forest','Preston','Powys','South Ribble','Wyre','Rossendale',
    'Darlington','Stockton-on-Tees','Burnley','Craven','Christchurch',
    'East Dorset','Weymouth and Portland','Middlesbrough','Conwy',
    'Southampton','Blackpool','Fareham','Eastleigh','Lancaster','Purbeck',
    'Gosport','Poole','Bournemouth','Wrexham','Pembrokeshire','Sedgemoor',
    'Cornwall','Taunton Deane','Halton','Mansfield','Wirral',
    'Carmarthenshire','Blackburn with Darwen','East Staffordshire',
    'Ceredigion','Richmond upon Thames','Isles of Scilly','Derbyshire Dales',
    'Adur','Rhondda Cynon Taf','Caerphilly','Ribble Valley',
    'The Vale of Glamorgan','Bridgend','Neath Port Talbot','Merthyr Tydfil',
    'Swansea','Cardiff','Blaenau Gwent','Newport','Hyndburn','Torfaen',
    'East Hertfordshire','Stevenage','Uttlesford','St Albans','East Lindsey',
    'Colchester','East Northamptonshire','Central Bedfordshire','Babergh',
    'Braintree','Huntingdonshire','Suffolk Coastal','Ipswich','Peterborough',
    'South Holland','Welwyn Hatfield','Boston','Watford','Maldon',
    'South Cambridgeshire','Hertsmere','Three Rivers','Bedford','Brentwood',
    'North Hertfordshire','Cambridge','Basildon','Harlow','Chelmsford',
    'Tendring','Dacorum','Luton','Harrow','Havering','Islington','Brent',
    'Greenwich','Castle Point','Aylesbury Vale','Milton Keynes','Kettering',
    'Thurrock','Westminster','Camden','Hammersmith and Fulham',
    'Kensington and Chelsea','Teignbridge','Southend-on-Sea','Hart',
    'East Hampshire','Lewisham','Rochford','Wandsworth',
    'North East Lincolnshire','Wokingham','Barking and Dagenham','Newham',
    'Portsmouth','Tower Hamlets','Daventry','Corby','Rushmoor',
    'South Northamptonshire','Wellingborough','Hackney','Torbay',
    'Northampton','Havant','Reading','Swale','Shepway','Hastings','Bromley',
    'Tunbridge Wells','Maidstone','Cherwell','Oadby and Wigston','Rother',
    'Canterbury','Ashford','Thanet','Dover','Nuneaton and Bedworth']


# Training constants
metric_dict = {
    'total_loss': 'Total Loss',
    'recon_loss': 'Reconstruction Loss',
    'kl_loss': 'KL Divergence Loss',
    'train_auc': 'Transductive AUROC',
    'train_ap': 'Transductive Average Precision',
    'test_auc': 'Inductive AUROC',
    'test_ap': 'Inductive Average Precision'
}

default_model = {
    'out_channels': 10,
    'model_type': 'gain',
    'num_layers': 2
}

GLOBAL_THRESHOLD = 0.6

# Filenames of saved datasets
saved_data_files = {
    ('No Bounds', "['metres', 'choice2km', 'nodecount2km', 'integration2km', 'choice10km', 'nodecount10km', 'integration10km', 'choice100km', 'nodecount100km', 'integration100km', 'choice2kmrank', 'choice10kmrank', 'integration10kmrank', 'integration2kmrank', 'choice2kmlog', 'choice10kmlog', 'choice100kmlog']", 'accident_count', 'dual', True, True, 50, False): 'clean_accident_full_pyg.pt',
    ('No Bounds', "['metres', 'choice2km', 'nodecount2km', 'integration2km', 'choice10km', 'nodecount10km', 'integration10km', 'choice100km', 'nodecount100km', 'integration100km', 'choice2kmrank', 'choice10kmrank', 'integration10kmrank', 'integration2kmrank', 'choice2kmlog', 'choice10kmlog', 'choice100kmlog']", 'accident_count', 'dual', True, True, 10, False): 'clean_accident_10m_full_pyg.pt',
    ('No Bounds', "['metres', 'choice2km', 'nodecount2km', 'integration2km', 'choice10km', 'nodecount10km', 'integration10km', 'choice100km', 'nodecount100km', 'integration100km', 'choice2kmrank', 'choice10kmrank', 'integration10kmrank', 'integration2kmrank', 'choice2kmlog', 'choice10kmlog', 'choice100kmlog', 'meridian_class']", 'accident_count', 'dual', False, True, 15, False): 'raw_accident_15m_full_pyg.pt',
    ('No Bounds', "['metres', 'choice2km', 'nodecount2km', 'integration2km', 'choice10km', 'nodecount10km', 'integration10km', 'choice100km', 'nodecount100km', 'integration100km', 'choice2kmrank', 'choice10kmrank', 'integration10kmrank', 'integration2kmrank', 'choice2kmlog', 'choice10kmlog', 'choice100kmlog']", 'accident_count', 'dual', True, False, 50, False): 'clean_accident_full_unconn_pyg.pt',
    ('No Bounds', "['metres', 'choice2km', 'nodecount2km', 'integration2km', 'choice10km', 'nodecount10km', 'integration10km', 'choice100km', 'nodecount100km', 'integration100km', 'choice2kmrank', 'choice10kmrank', 'integration10kmrank', 'integration2kmrank', 'choice2kmlog', 'choice10kmlog', 'choice100kmlog']", 'meridian_class', 'dual', True, True, 50, False): 'clean_meridian_full_pyg.pt',
    ('No Bounds', "['metres', 'choice2km', 'nodecount2km', 'integration2km', 'choice10km', 'nodecount10km', 'integration10km', 'choice100km', 'nodecount100km', 'integration100km', 'choice2kmrank', 'choice10kmrank', 'integration10kmrank', 'integration2kmrank', 'choice2kmlog', 'choice10kmlog', 'choice100kmlog']", 'meridian_class', 'dual', True, False, 50, False): 'clean_meridian_full_unconn_pyg.pt',
    ('No Bounds', "['metres', 'choice2km', 'nodecount2km', 'integration2km', 'choice10km', 'nodecount10km', 'integration10km', 'choice100km', 'nodecount100km', 'integration100km', 'choice2kmrank', 'choice10kmrank', 'integration10kmrank', 'integration2kmrank', 'choice2kmlog', 'choice10kmlog', 'choice100kmlog']", 'meridian_class', 'dual', False, True, 50, False): 'raw_meridian_full_conn_pyg.pt',
}