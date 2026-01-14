package excel

import (
	"fmt"
	"pos-distance/internal/models"
	"strconv"
	"strings"

	"github.com/xuri/excelize/v2"
)

func parseCoord(val string) (float64, error) {
	// Replace comma with dot for Turkish locales
	val = strings.TrimSpace(strings.ReplaceAll(val, ",", "."))
	if val == "" {
		return 0, fmt.Errorf("empty")
	}
	return strconv.ParseFloat(val, 64)
}

func OpenFile(filename string) (*excelize.File, error) {
	return excelize.OpenFile(filename)
}

func ReadSheet(f *excelize.File, sheetName string) ([]models.Customer, error) {
	rows, err := f.GetRows(sheetName)
	if err != nil {
		return nil, err
	}

	var customers []models.Customer
	// Assume header is row 0, start from row 1
	for i, row := range rows {
		if i == 0 {
			continue // Skip header
		}
		if len(row) < 11 {
			continue // Not enough columns
		}

		// J (index 9) -> Lat, K (index 10) -> Lon
		latStr := row[9]
		lonStr := row[10]

		lat, err1 := parseCoord(latStr)
		lon, err2 := parseCoord(lonStr)

		if err1 != nil || err2 != nil {
			continue // Skip invalid rows
		}

		c := models.Customer{
			ID:   row[0],
			Name: row[1],
			Loc: models.Coordinate{
				Lat: lat,
				Lon: lon,
			},
			RowIndex: i + 1,
		}
		customers = append(customers, c)
	}
	return customers, nil
}

func WriteResult(path string, data []models.ResultRow, sheetName string) error {
	f := excelize.NewFile()
	index, err := f.NewSheet(sheetName)
	if err != nil {
		return err
	}
	
	// Use Stream Writer for performance
	sw, err := f.NewStreamWriter(sheetName)
	if err != nil {
		return err
	}

	// Set header
	headers := []interface{}{
		"KACC Musteri No", "KACC Musteri Adı", "KACC Lat", "KACC Lon",
		"POS Musteri No", "POS Musteri Adı", "POS Lat", "POS Lon",
		"Mesafe (m)",
	}
	
	if err := sw.SetRow("A1", headers); err != nil {
		return err
	}

	// Write data in chunks
	for i, r := range data {
		rowNum := i + 2
		cell, _ := excelize.CoordinatesToCellName(1, rowNum)
		row := []interface{}{
			r.KaccID, r.KaccName, r.KaccLat, r.KaccLon,
			r.PosID, r.PosName, r.PosLat, r.PosLon,
			r.Distance,
		}
		if err := sw.SetRow(cell, row); err != nil {
			return err
		}
	}

	if err := sw.Flush(); err != nil {
		return err
	}

	f.SetActiveSheet(index)
	// Delete default sheet if exists
	if sheetName != "Sheet1" {
		f.DeleteSheet("Sheet1")
	}
	
	return f.SaveAs(path)
}
