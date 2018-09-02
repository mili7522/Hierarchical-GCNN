import pandas as pd
import os

os.chdir('Data')


# Bridges identified using http://stat.abs.gov.au/itt/r.jsp?ABSMaps

SA1s = pd.read_csv('Geography/SA1_2016_AUST.csv')
SA1s.set_index('SA1_MAINCODE_2016', inplace = True)

nswBridges = [(11703133752, 12104141710),
             (12104141625, 11703133750),
             (12104141510, 12201141820),
             (12203143102, 12202142336),
             (12203143151, 12202142332),
             (12102140413, 10201103012),
             (10202105329, 10202105321),
             (10202104846, 10202105417),
             (11101121207, 11101120701),
             (11102121903, 11102121503),
             (10804116402, 10804116501),
             (10804116501, 10401108030),
             (11201123646, 11201123626),
             (11201123614, 11201123611),
             (11203125019, 11203155014),
             (11203155203, 11203125430),
             (11703133450, 12002138909),
             (12002138921, 12001138536),
             (12001138506, 12602149808),
             (12602149808, 12602149904),
             (12602149904, 12101140056),
             (12001138426, 12602159108),
             (12001138458, 12501147312),
             (12501147306, 12502147718),
             (12801153123, 11904138224),
             (12802153556, 12802153823),
             (12802153423, 11901135835),
             (11901136015, 11901135837),
             (10703114332, 10701154701),
             (10104102727, 10104102728)]

nswBridges = [(SA1s.loc[a]['SA1_7DIGITCODE_2016'], SA1s.loc[b]['SA1_7DIGITCODE_2016']) for a,b in nswBridges]

nsw_neighbours = pd.read_csv('Geography/SA1_2016_NEIGHBOURS_expanded.csv', index_col = 0, names = ['src_SA1_7DIG16', 'nbr_SA1_7DIG16'], header = 0)

additional_nsw_neighbours = pd.DataFrame(nswBridges, columns = ['src_SA1_7DIG16', 'nbr_SA1_7DIG16'])

nsw_neighbours_with_bridges = pd.concat((nsw_neighbours, additional_nsw_neighbours), ignore_index = True)
nsw_neighbours_with_bridges[['src_SA1_7DIG16', 'nbr_SA1_7DIG16']].to_csv('Geography/2018-09-01-NSW-Neighbouring_Suburbs_With_Bridges.csv', index = False)