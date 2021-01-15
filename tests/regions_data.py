import gnocis as nc

rsA = nc.regions( 'A', [
	nc.region('X', 20, 50), nc.region('X', 80, 100), nc.region('X', 150, 300), nc.region('X', 305, 400), nc.region('X', 500, 600),
	nc.region('Y', 40, 100), nc.region('Y', 120, 200)
	] )
#
rsB = nc.regions( 'B', [
	nc.region('X', 30, 40), nc.region('X', 90, 120), nc.region('X', 140, 150), nc.region('X', 300, 310),
	nc.region('Y', 40, 100), nc.region('Y', 130, 300), nc.region('Y', 600, 700)
	] )

