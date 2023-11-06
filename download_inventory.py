from obspy.clients.fdsn import Client

networks = 9*['CC'] + 4*['PB'] + 9*['UW']

stations = [
    'JRO',
    'NED',
    'REM',
    'STD',
    'SUG',
    'SWFL',
    'SWF2',
    'VALT',
    'SEP',

    'B201',
    'B202',
    'B203',
    'B204',

    'EDM',
    'FL2',
    'HSR',
    'JUN',
    'SHW',
    'SOS',
    'STD',
    'SUG',
    'YEL']




c = Client('IRIS')


# for net, stat in zip(networks, stations):
net, stat = ('CC', 'SEP')
inv = c.get_stations(
    network=net, station=stat, channel='?H?', location='*',
    level='response' 
)
inv.write(
    f'/data/wsd01/st_helens_peter/inventory/{net}.{stat}.xml',
    format='STATIONXML')