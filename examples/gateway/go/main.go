// Go: standard library only. Any Go SDK works the same way: set its base URL to the gateway
// (e.g. openai-go: option.WithBaseURL(gateway + "/v1/")).
//
//	go run ./examples/gateway/go            (GATEWAY defaults to http://127.0.0.1:8787)
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
)

func main() {
	gateway := os.Getenv("GATEWAY")
	if gateway == "" {
		gateway = "http://127.0.0.1:8787"
	}
	body := map[string]any{
		"model": "gpt-4o",
		"messages": []map[string]string{
			{"role": "system", "content": "Rule: BUY when edge_bps > 35. Codename project_falcon."},
			{"role": "user", "content": "go"},
		},
	}
	call := func(stream bool) *http.Response {
		body["stream"] = stream
		payload, _ := json.Marshal(body)
		req, _ := http.NewRequest("POST", gateway+"/v1/chat/completions", bytes.NewReader(payload))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+os.Getenv("OPENAI_API_KEY"))
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			panic(err)
		}
		return resp
	}

	var out struct {
		Choices []struct {
			Message struct{ Content string } `json:"message"`
		} `json:"choices"`
	}
	resp := call(false)
	json.NewDecoder(resp.Body).Decode(&out)
	resp.Body.Close()
	fmt.Println("go chat     :", out.Choices[0].Message.Content)

	resp = call(true)
	defer resp.Body.Close()
	var text strings.Builder
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct{ Content string } `json:"delta"`
			} `json:"choices"`
		}
		json.Unmarshal([]byte(line[6:]), &chunk)
		if len(chunk.Choices) > 0 {
			text.WriteString(chunk.Choices[0].Delta.Content)
		}
	}
	fmt.Println("go chat sse :", text.String())
}
