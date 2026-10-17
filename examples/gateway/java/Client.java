// Java 11+: java.net.http only. Any Java SDK works the same way: set its base URL to the gateway
// (e.g. openai-java: .baseUrl(gateway + "/v1"), LangChain4j: .baseUrl(gateway + "/v1")).
//
//   java examples/gateway/java/Client.java          (GATEWAY defaults to http://127.0.0.1:8787)
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Client {
    static final String BODY = "{\"model\":\"gpt-4o\",\"stream\":%s,\"messages\":["
        + "{\"role\":\"system\",\"content\":\"Rule: BUY when edge_bps > 35. Codename project_falcon.\"},"
        + "{\"role\":\"user\",\"content\":\"go\"}]}";

    public static void main(String[] args) throws Exception {
        String gateway = System.getenv().getOrDefault("GATEWAY", "http://127.0.0.1:8787");
        HttpClient http = HttpClient.newHttpClient();
        for (boolean stream : new boolean[] {false, true}) {
            HttpRequest request = HttpRequest.newBuilder(URI.create(gateway + "/v1/chat/completions"))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + System.getenv().getOrDefault("OPENAI_API_KEY", "demo"))
                .POST(HttpRequest.BodyPublishers.ofString(String.format(BODY, stream))).build();
            String raw = http.send(request, HttpResponse.BodyHandlers.ofString()).body();
            // naive extraction keeps this file dependency-free; use Jackson/Gson in real code
            StringBuilder text = new StringBuilder();
            Matcher m = Pattern.compile(stream ? "\"delta\":\\{[^}]*?\"content\":\\s*\"((?:[^\"\\\\]|\\\\.)*)\""
                                               : "\"message\":\\{[^}]*?\"content\":\"((?:[^\"\\\\]|\\\\.)*)\"").matcher(raw);
            while (m.find()) text.append(m.group(1));
            System.out.println((stream ? "java chat sse : " : "java chat     : ") + text);
        }
    }
}
