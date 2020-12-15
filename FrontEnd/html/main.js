console.log("Connected!");


// document.getElementById("message");

document.getElementById("submissionButton").onclick = function () {
    
    console.log(document.getElementById("message").value)
    var settings = {
        "url": "http://127.0.0.1:5000/detectFake",
        "method": "POST",
        "timeout": 0,
        "headers": {
          "Content-Type": "application/json"
        },
        "data": JSON.stringify({"News": document.getElementById("message").value}),
      };
      
      $.ajax(settings).done(function (response) {
        console.log(response);
        alert(response.isFake)
      });
};