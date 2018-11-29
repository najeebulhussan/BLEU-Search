

function init() {
  console.log("checked");
  // Grab all category to the dropdown 
  Plotly.d3.csv("/static/js/Category/sub-art.csv", function(error, data) {
    function unpack(data, key) {
      return data.map(function(row) { return row[key]; });
    }
    art = unpack(data, 'Name');
    console.log(art);
    var selector = d3.select("#selDatasetart");
    // Use the list of coin names to populate the select options
    art.forEach((name) => {
      selector
        .append("option")
        .text(name)
        .property("value", name);
    });
  });
  
  Plotly.d3.csv("/static/js/Category/sub-business.csv", function(error, data) {
    function unpack(data, key) {
      return data.map(function(row) { return row[key]; });
    }
    art = unpack(data, 'Name');
    console.log(art);
    var selector = d3.select("#selDatasetbusiness");
    // Use the list of coin names to populate the select options
    art.forEach((name) => {
      selector
        .append("option")
        .text(name)
        .property("value", name);
    });
  });

  Plotly.d3.csv("/static/js/Category/sub-computer.csv", function(error, data) {
    function unpack(data, key) {
      return data.map(function(row) { return row[key]; });
    }
    art = unpack(data, 'Name');
    console.log(art);
    var selector = d3.select("#selDatasetcomputer");
    // Use the list of coin names to populate the select options
    art.forEach((name) => {
      selector
        .append("option")
        .text(name)
        .property("value", name);
    });
  });

  Plotly.d3.csv("/static/js/Category/sub-game.csv", function(error, data) {
    function unpack(data, key) {
      return data.map(function(row) { return row[key]; });
    }
    art = unpack(data, 'Name');
    console.log(art);
    var selector = d3.select("#selDatasetgame");
    // Use the list of coin names to populate the select options
    art.forEach((name) => {
      selector
        .append("option")
        .text(name)
        .property("value", name);
    });
  });

  Plotly.d3.csv("/static/js/Category/sub-health.csv", function(error, data) {
    function unpack(data, key) {
      return data.map(function(row) { return row[key]; });
    }
    art = unpack(data, 'Name');
    console.log(art);
    var selector = d3.select("#selDatasethealth");
    // Use the list of coin names to populate the select options
    art.forEach((name) => {
      selector
        .append("option")
        .text(name)
        .property("value", name);
    });
  });

  Plotly.d3.csv("/static/js/Category/sub-home.csv", function(error, data) {
    function unpack(data, key) {
      return data.map(function(row) { return row[key]; });
    }
    art = unpack(data, 'Name');
    console.log(art);
    var selector = d3.select("#selDatasethome");
    // Use the list of coin names to populate the select options
    art.forEach((name) => {
      selector
        .append("option")
        .text(name)
        .property("value", name);
    });
  });

  Plotly.d3.csv("/static/js/Category/sub-kidteen.csv", function(error, data) {
    function unpack(data, key) {
      return data.map(function(row) { return row[key]; });
    }
    art = unpack(data, 'Name');
    console.log(art);
    var selector = d3.select("#selDatasetkidteen");
    // Use the list of coin names to populate the select options
    art.forEach((name) => {
      selector
        .append("option")
        .text(name)
        .property("value", name);
    });
  });

  Plotly.d3.csv("/static/js/Category/sub-news.csv", function(error, data) {
    function unpack(data, key) {
      return data.map(function(row) { return row[key]; });
    }
    art = unpack(data, 'Name');
    console.log(art);
    var selector = d3.select("#selDatasetnews");
    // Use the list of coin names to populate the select options
    art.forEach((name) => {
      selector
        .append("option")
        .text(name)
        .property("value", name);
    });
  });

  Plotly.d3.csv("/static/js/Category/sub-recreation.csv", function(error, data) {
    function unpack(data, key) {
      return data.map(function(row) { return row[key]; });
    }
    art = unpack(data, 'Name');
    console.log(art);
    var selector = d3.select("#selDatasetrecreation");
    // Use the list of coin names to populate the select options
    art.forEach((name) => {
      selector
        .append("option")
        .text(name)
        .property("value", name);
    });
  });

  Plotly.d3.csv("/static/js/Category/sub-reference.csv", function(error, data) {
    function unpack(data, key) {
      return data.map(function(row) { return row[key]; });
    }
    art = unpack(data, 'Name');
    console.log(art);
    var selector = d3.select("#selDatasetreference");
    // Use the list of coin names to populate the select options
    art.forEach((name) => {
      selector
        .append("option")
        .text(name)
        .property("value", name);
    });
  });
  
  Plotly.d3.csv("/static/js/Category/sub-regional.csv", function(error, data) {
    function unpack(data, key) {
      return data.map(function(row) { return row[key]; });
    }
    art = unpack(data, 'Name');
    console.log(art);
    var selector = d3.select("#selDatasetregional");
    // Use the list of coin names to populate the select options
    art.forEach((name) => {
      selector
        .append("option")
        .text(name)
        .property("value", name);
    });
  });

  Plotly.d3.csv("/static/js/Category/sub-science.csv", function(error, data) {
    function unpack(data, key) {
      return data.map(function(row) { return row[key]; });
    }
    art = unpack(data, 'Name');
    console.log(art);
    var selector = d3.select("#selDatasetscience");
    // Use the list of coin names to populate the select options
    art.forEach((name) => {
      selector
        .append("option")
        .text(name)
        .property("value", name);
    });
  });

  Plotly.d3.csv("/static/js/Category/sub-shopping.csv", function(error, data) {
    function unpack(data, key) {
      return data.map(function(row) { return row[key]; });
    }
    art = unpack(data, 'Name');
    console.log(art);
    var selector = d3.select("#selDatasetshopping");
    // Use the list of coin names to populate the select options
    art.forEach((name) => {
      selector
        .append("option")
        .text(name)
        .property("value", name);
    });
  });

  Plotly.d3.csv("/static/js/Category/sub-society.csv", function(error, data) {
    function unpack(data, key) {
      return data.map(function(row) { return row[key]; });
    }
    art = unpack(data, 'Name');
    console.log(art);
    var selector = d3.select("#selDatasetsociety");
    // Use the list of coin names to populate the select options
    art.forEach((name) => {
      selector
        .append("option")
        .text(name)
        .property("value", name);
    });
  });

  Plotly.d3.csv("/static/js/Category/sub-sports.csv", function(error, data) {
    function unpack(data, key) {
      return data.map(function(row) { return row[key]; });
    }
    art = unpack(data, 'Name');
    console.log(art);
    var selector = d3.select("#selDatasetsports");
    // Use the list of coin names to populate the select options
    art.forEach((name) => {
      selector
        .append("option")
        .text(name)
        .property("value", name);
    });
  });

  Plotly.d3.csv("/static/js/Category/sub-world.csv", function(error, data) {
    function unpack(data, key) {
      return data.map(function(row) { return row[key]; });
    }
    art = unpack(data, 'Name');
    console.log(art);
    var selector = d3.select("#selDatasetworld");
    // Use the list of coin names to populate the select options
    art.forEach((name) => {
      selector
        .append("option")
        .text(name)
        .property("value", name);
    });
  });
  console.log(d3.select("#selDatasetart").property('value'));
  
  }
  
  function optionChanged() {
    
   
    // Fetch new data each time a new sample is selected
    let art =  d3.select("#selDatasetart").property('value');
    let business =  d3.select("#selDatasetbusiness").property('value');
    let computer =  d3.select("#selDatasetcomputer").property('value');
    let game =  d3.select("#selDatasetgame").property('value');
    let health =  d3.select("#selDatasethealth").property('value');
    let home =  d3.select("#selDatasethome").property('value');
    let kidteen =  d3.select("#selDatasetkidteen").property('value');
    let news =  d3.select("#selDatasetnews").property('value');
    let recreation =  d3.select("#selDatasetrecreation").property('value');
    let reference =  d3.select("#selDatasetreference").property('value');
    let regional =  d3.select("#selDatasetregional").property('value');
    let science =  d3.select("#selDatasetscience").property('value');
    let shopping =  d3.select("#selDatasetshopping").property('value');
    let society =  d3.select("#selDatasetsociety").property('value');
    let sports =  d3.select("#selDatasetsports").property('value');
    let world =  d3.select("#selDatasetworld").property('value');
    
    console.log(d3.select("#selDatasetshopping").property('value'));

    var current = [];

    if (art !=""){
      current.push("Category//art//"+art+".csv");
    }
    if  (business !="") {
      current.push("Category//business//"+business+".csv");
    }
    if  (computer !="") {
      current.push("Category//computer//"+computer+".csv");
    }
    if  (game !="") {
      current.push("Category//game//"+game+".csv");
    }
    if  (health !="") {
      current.push("Category//health//"+health+".csv");
    }
    if  (home !="") {
      current.push("Category//home//"+home+".csv");
    }
    if  (kidteen !="") {
      current.push("Category//kidteen//"+kidteen+".csv");
    }
    if  (news !="") {
      current.push("Category//news//"+news+".csv");
    }
    if  (recreation !="") {
      current.push("Category//recreation//"+recreation+".csv");
    }
    if  (reference !="") {
      current.push("Category//reference//"+reference+".csv");
    }
    if  (regional !="") {
      current.push("Category//regional//"+regional+".csv");
    }
    if  (science !="") {
      current.push("Category//science//"+science+".csv");
    }
    if  (shopping !="") {
      current.push("Category//shopping//"+shopping+".csv");
    }
    if  (society !="") {
      current.push("Category//society//"+society+".csv");
    }
    if  (sports !="") {
      current.push("Category//sports//"+sports+".csv");
    }
    if  (world !="") {
      current.push("Category//world//"+world+".csv");
    }

    if (current.length ==0) {
      current = ["Category//news//US.csv"];
    }
    console.log(current);
   return current
   // console.log("Category//art//"+art+".csv");
   // let time =  d3.select("#selDataset2").property('value');
    //buildplot(newcoin,curr,time);
  }

  function results(){
    
    //event.preventDefault();
    $("#voice").show();
    $("#analysis").show();
    d3.select("#search").html("");
    d3.select("#photos").html("");
    d3.select("#result").html("");
   $.get('/data', function(data) { 
    d3.select("#result").html(data.length + "  " + "search results");
        var source = [];
        for(let i = 0; i<data.length;i++) { 
          source.push({"seq":i+1,"title":data[i].Title,"link":data[i].Link,"description":data[i].Description,"image":data[i].Image,"keyword":data[i].Keyword});
        } 
        console.log(source)
        var svg = d3.select("#search");
        var chartGroup = svg.append("g")
        var group =  chartGroup.selectAll("div").data(source);
        var g = group.enter().append("g")
     g.append("div")
          .attr("id","line")
          .style("color", "black")
          .text(d=>d.seq + "  -  " + d.description +  "   " + "   " )
          .append("br");
     g.append("div")
          .style("color", "blue")
          .append("a")
          .text(d=> d.link)
          .attr("xlink:href", d => d.link)
          .on("click", function(d){
            window.open(d.link)});
    g.append("div")
    .style("color", "lightblue")
    .text(d=> "Keywords :" + "   " + "'" + d.keyword +"'");
    g.append('img')
     .attr('src', d=> d.image).filter(d=>d.image !="")
     .attr('height',"20%")
     .attr('width',"30%");
    g.append("br");
    g.append("label")
     .attr("for",function(d,i) { return 'a'+i; })
     .text("Analysis and Notes " + '\u00A0 \u00A0 \u00A0')
    g.append("input")
     .style("color", "blue")
     .attr("class","checkURL")
     .attr("unchecked", true)
     .attr("type", "checkbox")
     .attr("value", d => d.link)
     .attr("id", function(d,i) { return 'a'+i; });
    });
    }
   
    function voice(){
      $.ajax({
        dataType: 'json',
        url: '/voice',
        type: 'POST',
        contentType: 'application/json',
        success: function (result) {
            alert(result);
        },
        failure: function (errMsg) {
            alert(errMsg);
        }
      });
    }

    //Image search 
    function resultimages(){
    
      event.preventDefault();
      $("#voice").hide();
      $("#analysis").hide();
      d3.select("#search").html("");
      d3.select("#photos").html("");
      d3.select("#result").html("");
     $.get('/imagedata', function(data) { 
      d3.select("#result").html(data.length + "  " + "search results");
          var source = [];
          for(let i = 0; i<data.length;i++) { 
            source.push({"seq":i+1,"Link":data[i].Link,"Desc":data[i].Description,"Image":data[i].Image});
          } 
          console.log(source)
          var svg = d3.select("#search");
          var chartGroup = svg.append("g");
          var group =  chartGroup.selectAll("div").data(source);
          var g = group.enter().append("g")
      g.append('div')
       .attr("class","row")
       .append('div')
       .attr("class", "col-md-3")
       .append('img')
       .attr('src', d=> d.Image).filter(d=>d.image !="")
       .attr('height',"20%")
       .attr('width',"30%");
      g.append("br");
      });
      }
     
    function analyst(){
      //$("html").load("/analysis");
      d3.select("#ana").html("checked completed");
      
      var checkedValue = null;
      var checkvalues = []; 
      var inputElements = document.getElementsByClassName('checkURL');
      for(var i=0; inputElements[i]; ++i){
        if(inputElements[i].checked){
          checkedValue = inputElements[i].value;
          checkvalues.push(checkedValue);
        }}

      $.ajax({
        data: JSON.stringify(checkvalues),
        dataType: 'json',
        url: '/analysisinput',
        type: 'POST',
        contentType: 'application/json',
        success: function(response) {
          if (response != null) {
            //console.log(response);
           // d3.select("#analyst").html("checked completed 1");
           
            window.open("/analysis");
           // d3.select("#ana").html("checked completed");
        }
        else {
          d3.select("#analyst").html("");
          d3.select("#analyst").append("div")
            .attr("align","center")
            .html("No Data to analysis");
        }
      },
      error: function(err) {
          console.log(err);
      }
    });
      /*
      var checkedValue = null; 
      var inputElements = document.getElementsByClassName('checkURL');
      for(var i=0; inputElements[i]; ++i){
        if(inputElements[i].checked){
          checkedValue = inputElements[i].value;
          window.alert(checkedValue);
      }
       }
       */
      }

    function datatopython(){
      //check = d3.select("#searchoption").property("value");
    //  window.alert(d3.select("#bleusearch").property("value"));
     /// if (check ==="textsearch") {
      
     // }
      
      $("#loaddata").show();
      $("#voice").hide();
      $("#analysis").hide();
      
      d3.select("#result").html("");
      d3.select("#photos").html("");
     
      d3.select("#search").html("");
      
      curr =  optionChanged();
      console.log(curr);
      search = d3.select("#bleusearch").property('value');

      var para = [{
        'file': curr,
        'searchstring': search,
      }];
     console.log(para);
        // ajax the JSON to the server
        d3.select("#search").html("Searching...");
        let searchopt =  d3.select("#searchoption").property('value');
        if (searchopt == "textsearch") {
        $.ajax({
          data: JSON.stringify(para),
          dataType: 'json',
          url: '/receiver',
          type: 'POST',
          contentType: 'application/json',
          success: function(response) {
            
           
            if (response != null) {
              sessionStorage.setItem("checkingtext",d3.select("#bleusearch").property("value"));
            results(); 
          }
          else {
            $("#voice").hide();
            $("#analysis").hide();
            d3.select("#search").html("");
            d3.select("#search").append("div")
              .attr("align","center")
              .html("Search Not Found");
          }
            //print(response);
            //var data = $.parseJSON(JSON.stringify(response));
        },
        error: function(err) {
            console.log(err);
        }
           
            //event.preventDefault();
            /*var source = [];
            for(let i = 0; i<data.length;i++) { 
              source.push({"title":data[i].Title,"link":data[i].Link,"description":data[i].Description,"keyword":data[i].Keyword});
            } 
            var svg = d3.select("#search");
            var chartGroup = svg.append("g");
            console.log(source);
            chartGroup.selectAll("div")
              .data(source)
              .enter()
              .append("div")
              .text(d=> d.Title +  "   " + "_@_" + "   " )
              .append("a")
              .text(d=> d.Link)
              .attr("xlink:href", d => d.Link)
              .on("click", function(d){
                window.open(d.Link)})
                */
         
      });
    }
    else if (searchopt=="imagesearch") {
      $.ajax({
        data: JSON.stringify(para),
        dataType: 'json',
        url: '/receiverimage',
        type: 'POST',
        contentType: 'application/json',
        success: function(response) {
          if (response != null) {
            sessionStorage.setItem("checkingimage",d3.select("#bleusearch").property("value"));
            gridData();
        }
        else {
          $("#voice").hide();
          $("#analysis").hide();
          d3.select("#search").html("");
          d3.select("#search").append("div")
            .attr("align","center")
            .html("Search Not Found");
        }
          //print(response);
          //var data = $.parseJSON(JSON.stringify(response));
      },
      error: function(err) {
          console.log(err);
      }
    });
    }
  }

  init();
//Reload in case of refresh:
let searchopt =  d3.select("#searchoption").property('value');
let searchstr = document.getElementById("bleusearch").value;
if (searchopt == "textsearch" & searchstr !="") {
  document.getElementById("bleusearch").value= sessionStorage.getItem("checkingtext");
  results();
}
else if (searchopt == "imagesearch" & searchstr !="") {
  document.getElementById("bleusearch").value= sessionStorage.getItem("checkingimage");
  gridData();
}
//Checking
 function gridData() {

  $("#voice").hide();
  $("#analysis").hide();
  d3.select("#search").html("");
  d3.select("#result").html("");  
  d3.selectAll("#photos").remove();
  d3.selectAll("#phototable").remove();
  /*
	var source = new Array();
	var xpos = 1; //starting xpos and ypos at 1 so the stroke will show when we make the grid below
	var ypos = 1;
	var width = 50;
  var height = 50;
  var width1 = 50;
	var height1 = 50;
  var click = 0;
  var index = 0;
	$.get('/imagedata', function(data) { 
   // window.alert(data.length);

   // window.alert(parseInt(data.length/5));
        
         // source.push({"seq":i+1,"Link":data[i].Link,"Desc":data[i].Description,"Image":data[i].Image});
          // iterate for rows	
          for (var row = 0; row < parseInt(data.length/5); row++) {
            source.push( new Array());
		
		// iterate for cells/columns inside rows
            for (var column = 0; column < 5; column++) {
              source[row].push({
                x: xpos,
                y: ypos,
                width: width,
                height: height,
                click: click,
                image: data[index].Image
              })
			// increment the x position. I.e. move it over by 50 (width variable)
            xpos += width;
            index = column+1;
		        }
		// reset the x position after a row is complete
		       xpos = 1;
		// increment the y position for the next row. Move it down 50 (height variable)
	      	 ypos += height;	
         }
console.log(source);
  var grid = d3.select("#grid")
	.append("svg")
	.attr("width","1098")
	.attr("height","1098");
	
var row = grid.selectAll(".row")
	.data(source)
	.enter().append("g")
	.attr("class", "row");
	
  var column = row.selectAll("div")
	.data(function(d) { return d; })
	.enter().append("svg:image")
	.attr("x", function(d) { return d.x; })
	.attr("y", function(d) { return d.y; })
	.attr("width", function(d) { return d.width; })
  .attr("height", function(d) { return d.height; })
  .attr("xlink:href", d => d.image)
	.style("fill", "#fff")
	.style("stroke", "#222")
	.on('click', function(d) {
       d.click ++;
       if ((d.click)%4 == 0 ) { d3.select(this).style("fill","#fff"); }
	   if ((d.click)%4 == 1 ) { d3.select(this).style("fill","#2C93E8"); }
	   if ((d.click)%4 == 2 ) { d3.select(this).style("fill","#F56C4E"); }
	   if ((d.click)%4 == 3 ) { d3.select(this).style("fill","#838690"); }
    })
    
   
  });
  */
 var body = document.getElementsByTagName("body")[0];
 var tbl = document.createElement("table");
 tbl.setAttribute("id","phototable");
 var row = document.createElement("tr");
 var cell = document.createElement("td");
 var p = document.createElement("p");
 p.setAttribute("id","photos");
 p.setAttribute("align","center");
 $.get('/imagedata', function(data) { 
 for (var i=0, len = data.length; i < len; ++i) {
      var linkElement = document.createElement('a');
      linkElement.setAttribute("target", "_blank");
      linkElement.setAttribute("title",data[i].Description);
      linkElement.setAttribute("data-toggle","tooltip");
      linkElement.href = data[i].Link;
      var img = new Image();
      img.setAttribute("width","280");
      img.setAttribute("height","200");
      img.src = data[i].Image;
      
      linkElement.appendChild(img);
      p.appendChild(linkElement);

      cell.appendChild(p);
      row.appendChild(cell);
      tbl.appendChild(row);
      body.appendChild(tbl);
 }
});
}

