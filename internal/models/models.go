package models

type Coordinate struct {
	Lat float64
	Lon float64
}

type Customer struct {
	ID   string
	Name string
	Loc  Coordinate
	RowIndex int
}

type ResultRow struct {
	KaccID   string
	KaccName string
	KaccLat  float64
	KaccLon  float64
	PosID    string
	PosName  string
	PosLat   float64
	PosLon   float64
	Distance int
}
