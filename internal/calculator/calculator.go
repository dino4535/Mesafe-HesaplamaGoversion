package calculator

import (
	"fmt"
	"math"
	"pos-distance/internal/models"
	"runtime"
	"sync"
	"sync/atomic"
)

type ProgressCallback func(current, total int, msg string)
type LoggerCallback func(msg string)

func ComputeNearest(kaccList []models.Customer, posList []models.Customer, onProgress ProgressCallback, logger LoggerCallback) ([]models.ResultRow, error) {
	if len(kaccList) == 0 || len(posList) == 0 {
		return nil, fmt.Errorf("empty input lists")
	}

	total := len(kaccList)
	results := make([]models.ResultRow, total)

	numCPU := runtime.NumCPU()
	if numCPU < 1 {
		numCPU = 1
	}
	chunkSize := (total + numCPU - 1) / numCPU

	var wg sync.WaitGroup
	var processedCount int64 = 0

	logger(fmt.Sprintf("Starting parallel processing with %d CPUs, %d customers, %d POS", numCPU, len(kaccList), len(posList)))

	for i := 0; i < numCPU; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if start >= total {
			break
		}
		if end > total {
			end = total
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()

			for idx := s; idx < e; idx++ {
				source := kaccList[idx]
				var nearestIdx int = 0
				minDist := math.MaxFloat64

				// Find the nearest POS for this customer
				for pIdx, p := range posList {
					d := Haversine(source.Loc.Lat, source.Loc.Lon, p.Loc.Lat, p.Loc.Lon)
					if d < minDist {
						minDist = d
						nearestIdx = pIdx
					}
				}

				nearest := posList[nearestIdx]
				results[idx] = models.ResultRow{
					KaccID:   source.ID,
					KaccName: source.Name,
					KaccLat:  source.Loc.Lat,
					KaccLon:  source.Loc.Lon,
					PosID:    nearest.ID,
					PosName:  nearest.Name,
					PosLat:   nearest.Loc.Lat,
					PosLon:   nearest.Loc.Lon,
					Distance: int(math.Round(minDist)),
				}

				// Atomic increment for progress
				count := atomic.AddInt64(&processedCount, 1)
				if count%500 == 0 {
					if onProgress != nil {
						onProgress(int(count), total, "")
					}
				}
			}
		}(start, end)
	}

	wg.Wait()
	
	// Final progress update
	if onProgress != nil {
		onProgress(total, total, "")
	}
	
	logger("Calculation completed.")
	return results, nil
}

func ComputeRadius(kaccList []models.Customer, posList []models.Customer, radiusMeters float64, onProgress ProgressCallback, logger LoggerCallback) ([]models.ResultRow, error) {
	numCPU := runtime.NumCPU()
	total := len(kaccList)
	chunkSize := (total + numCPU - 1) / numCPU

	resultChan := make(chan []models.ResultRow, numCPU)
	var wg sync.WaitGroup

	logger(fmt.Sprintf("Starting Radius search (%.0fm) with %d CPUs", radiusMeters, numCPU))

	for i := 0; i < numCPU; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if start >= total {
			break
		}
		if end > total {
			end = total
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			var localRes []models.ResultRow

			for idx := s; idx < e; idx++ {
				src := kaccList[idx]

				for _, p := range posList {
					d := Haversine(src.Loc.Lat, src.Loc.Lon, p.Loc.Lat, p.Loc.Lon)
					if d <= radiusMeters {
						localRes = append(localRes, models.ResultRow{
							KaccID:   src.ID,
							KaccName: src.Name,
							KaccLat:  src.Loc.Lat,
							KaccLon:  src.Loc.Lon,
							PosID:    p.ID,
							PosName:  p.Name,
							PosLat:   p.Loc.Lat,
							PosLon:   p.Loc.Lon,
							Distance: int(math.Round(d)),
						})
					}
				}
			}
			resultChan <- localRes
		}(start, end)
	}

	go func() {
		wg.Wait()
		close(resultChan)
	}()

	var allResults []models.ResultRow
	processedChunks := 0

	for resChunk := range resultChan {
		allResults = append(allResults, resChunk...)
		processedChunks++
		if onProgress != nil {
			onProgress(processedChunks*chunkSize, total, "")
		}
	}

	logger("Radius calculation completed.")
	return allResults, nil
}
