function getBHKValue() {
    var uiBHK = document.getElementsByName("uiBHK");
    for (var i = 0; i < uiBHK.length; i++) {
        if (uiBHK[i].checked) {
            return parseInt(uiBHK[i].value);
        }
    }
    return -1; // Invalid Value
}

function getEMIValue() {
    var emi = document.getElementById("uiEMI").value;
    if (emi === "Others") {
        return document.getElementById("otherEMI").value;
    } else {
        console.log(emi);
        return emi;
    }
}

function onClickedEstimatePrice() {
    console.log("Estimate price button clicked");
    var sqft = document.getElementById("uiSqft").value;
    var bhk = getBHKValue();
    var location = document.getElementById("uiLocations").value;
    var emi = getEMIValue();
    var estPrice = document.getElementById("uiEstimatedPrice");

    var url = "http://127.0.0.1:5000/predict_home_price"; // Use this if you are NOT using nginx which is first 7 tutorials

    $.post(url, {
        location: location,
        total_sqft: parseInt(sqft),
        bhk: bhk,
        emi: emi
    }, function(data, status) {
        console.log(data.estimated_price);
        price=data.estimated_price/100000;
        estPrice.innerHTML = "<h2>Your estimated price is " + price.toString() + "Lakhs</h2>";
        console.log(status);
    });
}

function onPageLoad() {
    console.log("document loaded");
    var url = "http://127.0.0.1:5000/get_location_names"; // Use this if you are NOT using nginx which is first 7 tutorials

    $.get(url, function(data, status) {
        console.log("got response for get_location_names request");
        if (data) {
            var locations = data.locations;
            var uiLocations = document.getElementById("uiLocations");
            $('#uiLocations').empty();
            for (var i = 0; i < locations.length; i++) {
                var opt = new Option(locations[i]);
                uiLocations.appendChild(opt);
            }
        }
    });
}

window.onload = onPageLoad;