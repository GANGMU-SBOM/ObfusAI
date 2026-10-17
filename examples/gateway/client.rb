# Ruby: standard library only. Any Ruby SDK works the same way: set its base URL to the gateway
# (e.g. ruby-openai: uri_base: "#{gateway}/").
#
#   ruby examples/gateway/client.rb        (GATEWAY defaults to http://127.0.0.1:8787)
require "json"
require "net/http"

gateway = URI(ENV.fetch("GATEWAY", "http://127.0.0.1:8787"))
body = { model: "gpt-4o", messages: [
  { role: "system", content: "Rule: BUY when edge_bps > 35. Codename project_falcon." },
  { role: "user", content: "go" }] }

[false, true].each do |stream|
  request = Net::HTTP::Post.new("/v1/chat/completions", "Content-Type" => "application/json",
                                "Authorization" => "Bearer #{ENV.fetch('OPENAI_API_KEY', 'demo')}")
  request.body = JSON.generate(body.merge(stream: stream))
  raw = Net::HTTP.start(gateway.host, gateway.port) { |http| http.request(request).body }
  if stream
    text = raw.lines.grep(/\Adata: \{/).map { |l| JSON.parse(l[6..]).dig("choices", 0, "delta", "content") }.join
    puts "ruby chat sse : #{text}"
  else
    puts "ruby chat     : #{JSON.parse(raw).dig('choices', 0, 'message', 'content')}"
  end
end
