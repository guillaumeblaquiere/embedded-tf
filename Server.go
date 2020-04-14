package main

import (
	"bufio"
	"bytes"
	"cloud.google.com/go/storage"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/gorilla/mux"
	"google.golang.org/api/iterator"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

type tfAnswer struct {
	Prediction []interface{} `json:"predictions"`
	Error      string        `json:"error"`
}
type inputPrediction struct {
	Instances []interface{} `json:"instances"`
}

const (
	//Name of the model when tensorflow start
	MODEL_NAME = "mymodel"
	//Local storage of the model
	LOCAL_MODEL_PATH = "/tmp/model/"
	//Number of the model. Required by Tensorflow. The value doesn't matter here
	MODEL_DUMMY_VERSION = "000000/"

	//Local storage of the input file(s)
	LOCAL_INPUT_PATH = "/tmp/input/"
	//Local storage of the prediction output file(s)
	LOCAL_OUTPUT_PATH = "/tmp/output/"

	//The prefix of a GCS bucket definition
	BUCKET_PREFIX = "gs://"
	//The prefix of all generated prediction file(s)
	OUTPUT_PREFIX = "prediction_"

	//The API Rest port for Tensorflow server
	TF_PORT = "8501"
	//URL to call for a prediction on Tensorflow server
	TF_URL = "http://localhost:" + TF_PORT + "/v1/models/" + MODEL_NAME + ":predict"
	//Content type of the request to Tensorflow server
	TF_CONTENT_TYPE = "application/json"
	//The tensorflow server start timeout
	TF_TIMEOUT = 30
)

//Run the server on the default port.
func main() {
	router := initializeRouter()
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%s", port), router))
}

//Initialize the router. Only one route.
func initializeRouter() *mux.Router {
	// StrictSlash is true => redirect /cars/ to /cars
	router := mux.NewRouter().StrictSlash(true)

	router.Methods("GET").Path("/").HandlerFunc(LoadAndPredict)
	return router
}

//Extract the required bucket/path params from the Query parameters
func getParam(r *http.Request, paramName string) (string, string, error) {
	param, ok := r.URL.Query()[paramName]
	if !ok || len(param[0]) < 1 {
		return "", "", errors.New(fmt.Sprintf("Query Param '%s' is missing", paramName))
	}
	bucket, path, err := extractLocation(param[0])
	if err != nil {
		return "", "", errors.New(fmt.Sprintf("'%s' bad formatted: %s", paramName, err.Error()))
	}
	return bucket, path, nil

}

// Input File must be a JSONL (json line, with 1 full and consistent JSON object on one line)
func LoadAndPredict(w http.ResponseWriter, r *http.Request) {

	// Get Model param
	bucketModel, pathModel, err := getParam(r, "model")
	if err != nil {
		log.Println(err)
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintln(w, err.Error())
		return
	}

	// Model path must be the directory where the pb and variables are stored
	if !strings.HasSuffix(pathModel, "/") {
		pathModel += "/"
	}

	// Get Input param
	bucketInput, pathInput, err := getParam(r, "input")
	if err != nil {
		log.Println(err)
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintln(w, err.Error())
		return
	}

	// Get Output  param
	bucketOutput, pathOutput, err := getParam(r, "output")
	if err != nil {
		log.Println(err)
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintln(w, err.Error())
		return
	}

	log.Println("param parsed successfully. Start process")

	// Clear the previous execution
	os.RemoveAll(LOCAL_MODEL_PATH)
	os.RemoveAll(LOCAL_INPUT_PATH)
	os.RemoveAll(LOCAL_OUTPUT_PATH)

	//Create the storage client
	ctx := context.Background()
	client, err := storage.NewClient(ctx)
	if err != nil {
		log.Println(err)
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintln(w, "error when creating storage client")
		return
	}

	//Download model
	err = downloadFiles(ctx, client.Bucket(bucketModel), pathModel, LOCAL_MODEL_PATH+MODEL_DUMMY_VERSION)
	if err != nil {
		log.Println(err)
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintln(w, "error when downloading model files")
		return
	}

	log.Println("model loaded to " + LOCAL_MODEL_PATH + MODEL_DUMMY_VERSION)

	// Start tensorflow serving with the model
	cmd := exec.Command("tensorflow_model_server", "--port=8500", "--rest_api_port="+TF_PORT,
		"--model_name="+MODEL_NAME, "--model_base_path="+LOCAL_MODEL_PATH)

	// Blocking start until the initialization
	err = startAndWaitTF(cmd)
	defer cmd.Process.Kill()

	if err != nil {
		log.Println(err)
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintln(w, "error when starting tensorflow")
		return
	}

	// Download input
	// Copy a full directory if finish by /, else only the file
	if strings.HasSuffix(pathInput, "/") {
		//Copy all files and subdir
		err = downloadFiles(ctx, client.Bucket(bucketInput), pathInput, LOCAL_INPUT_PATH)
	} else {
		// Copy only the provided file name
		// Get the file name
		n := pathInput[strings.LastIndex(pathInput, "/")+1:]
		// Create the directory
		os.MkdirAll(LOCAL_INPUT_PATH, 0755)
		err = downloadFile(ctx, client.Bucket(bucketInput), pathInput, LOCAL_INPUT_PATH+n)
	}
	if err != nil {
		log.Println(err)
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintln(w, "error when downloading input files")
		return
	}

	log.Println("input loaded to " + LOCAL_INPUT_PATH)

	//Make the prediction with the inputs
	err = makePredictions(LOCAL_INPUT_PATH, LOCAL_OUTPUT_PATH)
	if err != nil {
		log.Println(err)
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintln(w, "error during predictions")
		return
	}

	log.Println("Predictions done")

	// Upload file to storage
	err = uploadFile(ctx, client.Bucket(bucketOutput), pathOutput, LOCAL_OUTPUT_PATH)

	log.Println("output uploaded")

	w.WriteHeader(http.StatusOK)
	fmt.Fprintln(w, "predictions completed")
}

//Start the Tensorflow server and wait the entry "Exporting HTTP/REST API" for considering the
//start completed and ready to use.
//If the server is not in a ready state after a timeout, an error is raised.
func startAndWaitTF(cmd *exec.Cmd) error {
	var stderr []byte
	var errStderr error
	cmd.Stdout = os.Stdout
	stderrIn, _ := cmd.StderrPipe()

	err := cmd.Start()
	if err != nil {
		return err
	}

	started := make(chan bool, 1)
	//Catch the output in a goroutine and evaluate them!
	go func() {
		stderr, errStderr = copyAndCapture(os.Stderr, stderrIn)
		if errStderr != nil {
			started <- false
		}
		started <- true
	}()

	// Wait, the TF startup or the timeout
	select {
	case res := <-started:
		if res {
			log.Println("Tensorflow Started. Continue the process")
			// Redirect the output
			cmd.Stderr = os.Stderr
		} else {
			return errStderr
		}
	case <-time.After(TF_TIMEOUT * time.Second):
		log.Printf("timeout exceeded. TF doesn't start in %d seconds\n", TF_TIMEOUT)
		return errors.New("timeout exceeded")
	}
	return nil
}

// Run TF and capture the output. Exit in success when "Exporting HTTP/REST API" is found in the logs
func copyAndCapture(w io.Writer, r io.Reader) ([]byte, error) {
	var out []byte
	buf := make([]byte, 1024, 1024)
	for {
		//Let the time to the buffer to contain char
		<-time.After(1 * time.Second)
		n, err := r.Read(buf[:])
		if n > 0 {
			d := buf[:n]
			out = append(out, d...)
			_, err := w.Write(d)
			if err != nil {
				return out, err
			}
			if strings.Contains(string(d), "Exporting HTTP/REST API") {
				// The server is running
				return out, nil
			}
		}
		if err != nil {
			return out, err
		}
	}
}

//Upload file from localPath into the path of the bucket. The bucket path must be a directory.
//All the files and subdirectory in the localPath are uploaded
func uploadFile(ctx context.Context, bucket *storage.BucketHandle, path string, localPath string) error {
	if !strings.HasSuffix(path, "/") {
		path += "/"
	}

	list, err := ioutil.ReadDir(localPath)
	if err != nil {
		return err
	}

	for _, file := range list {
		if file.IsDir() {
			err = uploadFile(ctx, bucket, path+file.Name()+"/", localPath+file.Name()+string(filepath.Separator))
			if err != nil {
				return err
			}
		} else {
			f, err := os.Open(localPath + file.Name())
			if err != nil {
				return err
			}
			defer f.Close()

			w := bucket.Object(path + file.Name()).NewWriter(ctx)
			if _, err = io.Copy(w, f); err != nil {
				return err
			}
			if err := w.Close(); err != nil {
				return err
			}
		}
	}
	return nil
}

//Request the prediction to Tensorflow server. Submit each file and subdirectory into inputPath
//and store the prediction into the outputPath, with the same folder structure as the inputPath
//The output file name have the same name as the input with predictions_ as prefix
func makePredictions(inputPath string, outputPath string) error {
	list, err := ioutil.ReadDir(inputPath)
	if err != nil {
		return err
	}

	// Make directory is required
	os.MkdirAll(outputPath, 0755)

	//Iterate over files
	for _, file := range list {
		if file.IsDir() {
			err = makePredictions(inputPath+file.Name()+string(filepath.Separator),
				outputPath+file.Name()+string(filepath.Separator))
			if err != nil {
				return err
			}
		} else {
			// Format input
			f, err := os.Open(inputPath + file.Name())
			if err != nil {
				return err
			}
			defer f.Close()

			finput, err := formatInput(f)
			if err != nil {
				return err
			}

			// Call TF for prediction
			resp, err := http.Post(TF_URL, TF_CONTENT_TYPE, strings.NewReader(finput))
			if err != nil {
				return err
			}

			// Write the output
			foutput, err := formatOutput(resp.Body)
			if err != nil {
				return err
			}

			o, err := os.Create(outputPath + OUTPUT_PREFIX + file.Name())
			if err != nil {
				return err
			}
			defer o.Close()

			_, err = io.Copy(o, strings.NewReader(foutput))
		}
	}
	return nil
}

//Format the output path as a JSON line format. Remove the "predictions" JSON array encapsulation of the
//Tensorflow server response body.
func formatOutput(input io.Reader) (string, error) {
	output, err := ioutil.ReadAll(input)
	if err != nil {
		return "", err
	}

	//Fix JSON unmarshal in case of error
	output = bytes.ReplaceAll(output, []byte("\\"), []byte("\\\\"))

	//Unmarshal the prediction JSON
	answer := tfAnswer{}
	err = json.Unmarshal(output, &answer)
	if err != nil {
		log.Printf("Error during answer unmarshal %s\n", output)
		return "", err
	}
	if answer.Error != "" {
		// Prediction error
		return "", errors.New(answer.Error)
	}

	// Read only the content and return it
	ret := ""
	for _, p := range answer.Prediction {
		b, err := json.Marshal(p)
		if err != nil {
			return "", err
		}
		ret += string(b) + "\n"
	}
	return ret, err
}

//Get the JSON line as input and format it as expected by Tensorflow server:
//Encapsulate the JSON line into a "intances" JSON array
func formatInput(input io.Reader) (string, error) {
	i := inputPrediction{Instances: []interface{}{}}
	scanner := bufio.NewScanner(input)
	for scanner.Scan() {
		var o interface{}
		err := json.Unmarshal(scanner.Bytes(), &o)
		if err != nil {
			return "", err
		}
		i.Instances = append(i.Instances, o)
	}
	if err := scanner.Err(); err != nil {
		return "", err
	}
	b, err := json.Marshal(i)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

//Extract the location from Param
func extractLocation(location string) (string, string, error) {
	if !strings.HasPrefix(location, BUCKET_PREFIX) {
		return "", "", errors.New("location must start with '" + BUCKET_PREFIX + "'")
	}
	location = location[len(BUCKET_PREFIX):]
	s := strings.SplitN(location, "/", 2)
	return s[0], s[1], nil
}

//Download files from storage to the localDest. If there is subdirectory into GCS path, a loop is performed for getting
//subdirectories
//The path must represent a GCS directory (prefix)
func downloadFiles(ctx context.Context, bucket *storage.BucketHandle, path string, localDest string) error {

	if !strings.HasSuffix(path, "/") {
		return errors.New("downloadFiles: path must be GCS directory")
	}

	// Make directory is required
	os.MkdirAll(localDest, 0755)

	// List all file in bucket
	it := bucket.Objects(ctx, &storage.Query{Prefix: path})
	for {
		attrs, err := it.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			return err
		}

		n := attrs.Name[len(path):]
		if n == "" {
			// Root path of the bucket filter path
			continue
		}
		if strings.HasSuffix(n, "/") {
			//it's a directory, loop on it
			err = downloadFiles(ctx, bucket, attrs.Name, localDest+n)
			if err != nil {
				return err
			}
		} else {
			err = downloadFile(ctx, bucket, attrs.Name, localDest+n)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

//Download a single file from storage to the localDest. The path must represent a GCS file
func downloadFile(ctx context.Context, bucket *storage.BucketHandle, path string, dest string) error {
	if strings.HasSuffix(path, "/") {
		return errors.New("downloadFiles: path must be GCS file")
	}

	//Copy each file in the dest directory
	src, err := bucket.Object(path).NewReader(ctx)
	if err != nil {
		return err
	}
	defer src.Close()

	destination, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer destination.Close()

	_, err = io.Copy(destination, src)
	if err != nil {
		return err
	}
	return nil
}
