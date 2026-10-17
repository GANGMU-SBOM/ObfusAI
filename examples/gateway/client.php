<?php
// PHP: curl only. Any PHP SDK works the same way: set its base URL to the gateway
// (e.g. openai-php/client: ->withBaseUri("$gateway/v1")).
//
//   php examples/gateway/client.php        (GATEWAY defaults to http://127.0.0.1:8787)
$gateway = getenv("GATEWAY") ?: "http://127.0.0.1:8787";
$messages = [
    ["role" => "system", "content" => "Rule: BUY when edge_bps > 35. Codename project_falcon."],
    ["role" => "user", "content" => "go"],
];
foreach ([false, true] as $stream) {
    $ch = curl_init("$gateway/v1/chat/completions");
    curl_setopt_array($ch, [
        CURLOPT_POST => true,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_HTTPHEADER => ["Content-Type: application/json", "Authorization: Bearer " . (getenv("OPENAI_API_KEY") ?: "demo")],
        CURLOPT_POSTFIELDS => json_encode(["model" => "gpt-4o", "stream" => $stream, "messages" => $messages]),
    ]);
    $raw = curl_exec($ch);
    curl_close($ch);
    if ($stream) {
        $text = "";
        foreach (explode("\n", $raw) as $line) {
            if (strpos($line, "data: {") === 0) {
                $chunk = json_decode(substr($line, 6), true);
                $text .= $chunk["choices"][0]["delta"]["content"] ?? "";
            }
        }
        echo "php chat sse  : $text\n";
    } else {
        echo "php chat      : " . json_decode($raw, true)["choices"][0]["message"]["content"] . "\n";
    }
}
