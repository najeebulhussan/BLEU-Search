function checkreturn (){
    window.alert("good morning");
    d3.select("#ana").html("checked completed");
}

function getanalyst (){
    $.ajaxPrefilter( function (options) {
        if (options.crossDomain && jQuery.support.cors) {
          var http = (window.location.protocol === 'http:' ? 'http:' : 'https:');
          options.url = http + '//cors-anywhere.herokuapp.com/' + options.url;
          //options.url = "http://cors.corsproxy.io/url=" + options.url;
        }
      });
      
      
    $.get("/analysisdata", function(data) { 
        var source = [];
       

        for(let i = 1; i<data.length;i++) { 
          source.push({"seq":i,"Content":data[i].Content,"Numbers":data[i].Numbers});
        } 
        console.log(source);
        var svg = d3.select("#analyst");
        var chartGroup = svg.append("g")
        var group =  chartGroup.selectAll("div").data(source);
        var g = group.enter().append("g")
        g.append("br");
        g.append("div")
        .attr("id","notes")
        .style("color", "black")
        .text(d=>"Notes" + " " + d.seq  + ":" + " " + d.Content)
        .append("br");
        g.append("br");
        g.append("div")
        .style("color", "blue")
        .text(d=> "Number(s)" +":" + " " + d.Numbers);
      
        d3.select("#title").html(data[0].Numbers);

        $.get(data[0].Content,function (response) {
              d3.select("#ana").html(response);
      });
    });
}

getanalyst();
