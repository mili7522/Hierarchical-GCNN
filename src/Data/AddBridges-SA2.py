import pandas as pd

# Bridges identified using http://stat.abs.gov.au/itt/r.jsp?ABSMaps

neighbours = [(117031337,121041417, 'Sydney - Haymarket - The Rocks', 'North Sydney - Lavender Bay'),
              (121041415, 122011418, 'Mosman', 'Balgowlah - Clontarf - Seaforth'),
              (121031408, 122031427, 'Lindfield - Roseville', 'Forestville - Killarney Heights'),
              (122031431, 122021423, 'Narrabeen - Collaroy', 'Warriewood - Mona Vale'),
              (121021404, 102011030, 'Berowra - Brooklyn - Cowan', 'Calga - Kulnura'),
              (120021389, 120011385, 'Lilyfield - Rozelle', 'Drummoyne - Rodd Point'),
              (120011385, 126021498, 'Drummoyne - Rodd Point', 'Gladesville - Huntleys Point'),
              (126021499, 121011400, 'Hunters Hill - Woolwich', 'Lane Cove - Greenwich'),
              (120011384, 126021591, 'Concord West - North Strathfield', 'Ryde'),
              (120011384, 125011473, 'Concord West - North Strathfield', 'Homebush Bay - Silverwater'),
              (128011531, 119041382, 'Sylvania - Taren Point', 'Sans Souci - Ramsgate'),
              (128011531, 119031374, 'Sylvania - Taren Point', 'South Hurstville - Blakehurst'),
              (128021536, 119031371, 'Oyster Bay - Como - Jannali', 'Oatley - Hurstville Grove'),
              (128021536, 128021535, 'Oyster Bay - Como - Jannali', 'Menai - Lucas Heights - Woronora'),
              (128021534, 119011358, 'llawong - Alfords Point', 'Padstow'),
              (119031372, 119011358, 'Peakhurst - Lugarno', 'Padstow')]


neighbouring_suburbs = pd.read_csv('Geography/2018-08-28-SA2Neighbouring_Suburbs.txt')

additional_neighbours = pd.DataFrame(neighbours, columns = ['src_SA2_MAIN16', 'nbr_SA2_MAIN16', 'src_SA2_NAME16', 'nbr_SA2_NAME16'])

additional_neighbours['OBJECTID'] = range(neighbouring_suburbs.iloc[-1]['OBJECTID'] + 1, neighbouring_suburbs.iloc[-1]['OBJECTID'] + 1 + len(additional_neighbours))
additional_neighbours['LENGTH'] = 1/100  # Set arbitary length of 10 m (bridge width)


neighbours_with_bridges = pd.concat((neighbouring_suburbs, additional_neighbours), ignore_index = True)

neighbours_with_bridges[['src_SA2_MAIN16', 'nbr_SA2_MAIN16', 'LENGTH']].to_csv('Geography/2018-08-28-SA2Neighbouring_Suburbs_With_Bridges.csv', index = False)
