package main

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"pos-distance/internal/calculator"
	"pos-distance/internal/excel"
	"pos-distance/internal/models"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-contrib/sessions"
	"github.com/gin-contrib/sessions/cookie"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// === Job System ===

type JobStatus string

const (
	StatusRunning JobStatus = "running"
	StatusDone    JobStatus = "done"
	StatusError   JobStatus = "error"
)

type JobResult struct {
	Mode     string `json:"mode"`
	Rows     int    `json:"rows"`
	Sheet    string `json:"sheet"`
	Output   string `json:"output"` // Full path
	Filename string `json:"filename"` // Just filename for download
}

type Job struct {
	ID        string
	Status    JobStatus
	Logs      []string
	Progress  int // 0-100
	Result    *JobResult
	Error     string
	CancelFn  func() // Function to cancel the context (not used heavily in v1 but good to have)
	Mutex     sync.RWMutex
	CreatedAt time.Time
}

var (
	JobStore = make(map[string]*Job)
	JobLock  sync.RWMutex
)

func NewJob() *Job {
	return &Job{
		ID:        uuid.New().String(),
		Status:    StatusRunning,
		Logs:      []string{},
		CreatedAt: time.Now(),
	}
}

func (j *Job) Log(msg string) {
	j.Mutex.Lock()
	defer j.Mutex.Unlock()
	ts := time.Now().Format("15:04:05")
	j.Logs = append(j.Logs, fmt.Sprintf("[%s] %s", ts, msg))
}

func (j *Job) SetProgress(current, total int, msg string) {
	j.Mutex.Lock()
	defer j.Mutex.Unlock()
	if total > 0 {
		j.Progress = int(float64(current) / float64(total) * 100)
	}
	if msg != "" {
		ts := time.Now().Format("15:04:05")
		j.Logs = append(j.Logs, fmt.Sprintf("[%s] %s", ts, msg))
	}
}

func GetJob(id string) *Job {
	JobLock.RLock()
	defer JobLock.RUnlock()
	return JobStore[id]
}

// === Main ===

const (
	LoginUser = "user"
	LoginPass = "Dino202545"
	SecretKey = "your-secret-key-here-for-pos-distance-app"
)

func main() {
	// Set Gin mode
	gin.SetMode(gin.ReleaseMode)

	r := gin.Default()

	// Setup Sessions
	store := cookie.NewStore([]byte(SecretKey))
	r.Use(sessions.Sessions("mysession", store))

	// Load Templates
	r.LoadHTMLGlob("templates/*")

	// Auth Middleware
	authRequired := func(c *gin.Context) {
		session := sessions.Default(c)
		user := session.Get("user")
		if user == nil {
			c.Redirect(http.StatusFound, "/login")
			c.Abort()
			return
		}
		c.Next()
	}

	// Routes
	r.GET("/login", func(c *gin.Context) {
		c.HTML(http.StatusOK, "login.html", gin.H{})
	})

	r.POST("/login", func(c *gin.Context) {
		username := c.PostForm("username")
		password := c.PostForm("password")

		if username == LoginUser && password == LoginPass {
			session := sessions.Default(c)
			session.Set("user", username)
			session.Save()
			c.Redirect(http.StatusFound, "/")
		} else {
			c.HTML(http.StatusOK, "login.html", gin.H{
				"Error": "Hatalı kullanıcı adı veya şifre",
			})
		}
	})

	r.GET("/logout", func(c *gin.Context) {
		session := sessions.Default(c)
		session.Clear()
		session.Save()
		c.Redirect(http.StatusFound, "/login")
	})

	// Protected Routes
	authorized := r.Group("/")
	authorized.Use(authRequired)
	{
		authorized.GET("/", func(c *gin.Context) {
			c.HTML(http.StatusOK, "index.html", gin.H{})
		})

		authorized.POST("/run", func(c *gin.Context) {
			file, err := c.FormFile("input_file")
			if err != nil {
				c.HTML(http.StatusOK, "index.html", gin.H{"Message": "Lütfen bir dosya seçin."})
				return
			}
			
			mode := c.PostForm("mode")
			metersStr := c.PostForm("meters")
			meters, _ := strconv.ParseFloat(metersStr, 64)
			
			// Save uploaded file
			os.MkdirAll("uploads", 0755)
			os.MkdirAll("output", 0755)
			
			inputPath := filepath.Join("uploads", fmt.Sprintf("%s_%s", uuid.New().String(), file.Filename))
			if err := c.SaveUploadedFile(file, inputPath); err != nil {
				c.HTML(http.StatusOK, "index.html", gin.H{"Message": "Dosya yüklenemedi."})
				return
			}

			// Create Job
			job := NewJob()
			JobLock.Lock()
			JobStore[job.ID] = job
			JobLock.Unlock()

			// Start Processing in Goroutine
			go processJob(job, inputPath, mode, meters)

			c.HTML(http.StatusOK, "index.html", gin.H{
				"JobID": job.ID,
				"Message": "İşlem başlatıldı...",
			})
		})

		authorized.GET("/logs", func(c *gin.Context) {
			jobID := c.Query("job_id")
			job := GetJob(jobID)
			if job == nil {
				c.JSON(http.StatusOK, gin.H{"ok": false, "error": "Job not found"})
				return
			}

			job.Mutex.RLock()
			logs := make([]string, len(job.Logs))
			copy(logs, job.Logs)
			status := job.Status
			progress := job.Progress
			job.Mutex.RUnlock()

			c.JSON(http.StatusOK, gin.H{
				"ok":       true,
				"logs":     logs,
				"status":   status,
				"progress": progress,
			})
		})

		authorized.GET("/status", func(c *gin.Context) {
			jobID := c.Query("job_id")
			job := GetJob(jobID)
			if job == nil {
				c.JSON(http.StatusOK, gin.H{"ok": false})
				return
			}
			job.Mutex.RLock()
			defer job.Mutex.RUnlock()
			
			res := gin.H{
				"ok":     true,
				"status": job.Status,
				"error":  job.Error,
			}
			if job.Result != nil {
				res["result"] = job.Result
			}
			c.JSON(http.StatusOK, res)
		})

		authorized.POST("/cancel", func(c *gin.Context) {
			jobID := c.Query("job_id")
			job := GetJob(jobID)
			if job != nil {
				job.Log("Kullanıcı tarafından iptal isteği...")
				// In a real sophisticated app we would use Context cancellation
				// For now just logging it, stopping hard calculation loop is tricky without polling context
			}
			c.JSON(http.StatusOK, gin.H{"ok": true})
		})

		authorized.GET("/download-template", func(c *gin.Context) {
			// Serve a default template if exists
			// "template.xlsx" might be in root or python_legacy
			// We check legacy if not in root
			if _, err := os.Stat("template.xlsx"); err == nil {
				c.File("template.xlsx")
			} else if _, err := os.Stat("python_legacy/template.xlsx"); err == nil {
				c.File("python_legacy/template.xlsx")
			} else {
				c.String(404, "Template not found")
			}
		})

		authorized.GET("/download-result/:filename", func(c *gin.Context) {
			filename := c.Param("filename")
			target := filepath.Join("output", filename)
			c.File(target)
		})
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "9595"
	}
	
	fmt.Printf("Go POS Server running on port %s\n", port)
	r.Run(":" + port)
}

func processJob(job *Job, inputPath string, mode string, meters float64) {
	defer func() {
		if r := recover(); r != nil {
			job.Mutex.Lock()
			job.Status = StatusError
			job.Error = fmt.Sprintf("Panic: %v", r)
			job.Mutex.Unlock()
		}
	}()

	job.Log(fmt.Sprintf("Dosya işleniyor: %s", filepath.Base(inputPath)))
	
	f, err := excel.OpenFile(inputPath)
	if err != nil {
		job.Mutex.Lock()
		job.Status = StatusError
		job.Error = fmt.Sprintf("Excel dosyası açılamadı: %v", err)
		job.Mutex.Unlock()
		return
	}
	defer f.Close()

	// Read KACC
	job.Log("KACC sayfası okunuyor...")
	kaccList, err := excel.ReadSheet(f, "KACC")
	if err != nil {
		failJob(job, fmt.Sprintf("KACC okuma hatası: %v", err))
		return
	}
	job.Log(fmt.Sprintf("%d adet müşteri (KACC) okundu.", len(kaccList)))

	// Read Pos
	job.Log("Pos sayfası okunuyor...")
	posList, err := excel.ReadSheet(f, "Pos")
	if err != nil {
		failJob(job, fmt.Sprintf("Pos okuma hatası: %v", err))
		return
	}
	job.Log(fmt.Sprintf("%d adet nokta (Pos) okundu.", len(posList)))

	// Compute
	var results []models.ResultRow
	
	progressCb := func(current, total int, msg string) {
		job.SetProgress(current, total, msg)
	}
	loggerCb := func(msg string) {
		job.Log(msg)
	}

	start := time.Now()

	if mode == "nearest" {
		job.Log("En yakın nokta hesaplanıyor (Mode: Nearest)...")
		results, err = calculator.ComputeNearest(kaccList, posList, progressCb, loggerCb)
	} else {
		job.Log(fmt.Sprintf("Yarıçap hesaplanıyor (Radius: %.0fm)...", meters))
		results, err = calculator.ComputeRadius(kaccList, posList, meters, progressCb, loggerCb)
	}

	if err != nil {
		failJob(job, fmt.Sprintf("Hesaplama hatası: %v", err))
		return
	}

	elapsed := time.Since(start)
	job.Log(fmt.Sprintf("Hesaplama tamamlandı. Süre: %s", elapsed))

	// Write Output
	outputPath := strings.Replace(inputPath, ".xlsx", fmt.Sprintf("_%s.xlsx", mode), 1)
	outputPath = strings.Replace(outputPath, "uploads", "output", 1) // Move to output folder
	
	job.Log("Sonuç dosyası yazılıyor...")
	err = excel.WriteResult(outputPath, results, "Sonuclar")
	if err != nil {
		failJob(job, fmt.Sprintf("Yazma hatası: %v", err))
		return
	}

	job.Mutex.Lock()
	job.Status = StatusDone
	job.Log("İşlem başarıyla tamamlandı.")
	job.Result = &JobResult{
		Mode:     mode,
		Rows:     len(results),
		Sheet:    "Sonuclar",
		Output:   outputPath,
		Filename: filepath.Base(outputPath),
	}
	job.Progress = 100
	job.Mutex.Unlock()
}

func failJob(job *Job, msg string) {
	job.Mutex.Lock()
	job.Status = StatusError
	job.Error = msg
	job.Logs = append(job.Logs, "[ERROR] "+msg)
	job.Mutex.Unlock()
}
