#!pip install -e /Users/rajrsingh/workspace/lib/pixiedust
import pixiedust
# import pandas as pd

homesdf = pixiedust.sampleData(6)
# homesdf = pd.read_csv("https://openobjectstore.mybluemix.net/misc/milliondollarhomes.csv")

roadslines = {
    "type": "FeatureCollection", 
    "features": [
        {
            "type": "Feature", 
            "properties": {}, 
            "geometry": {
                "type": "LineString", 
                "coordinates": [
                  [-71.0771656036377,42.364537198664884],
                  [-71.07780933380127,42.36133451106724],
                  [-71.07562065124512,42.359812384483625],
                  [-71.07557773590088,42.35610204645879]
                ]
            }
        }, 
        {
            "type": "Feature", 
            "properties": {
                "name": "Highway to the Danger Zone"
            }, 
            "geometry": {
                "type": "LineString", 
                "coordinates": [
                  [-71.09179973602294,42.35848049347556],
                  [-71.08287334442139,42.356419177928906],
                  [-71.07184410095215,42.35794138670829],
                  [-71.06772422790527,42.35686315929846]
                ]
            }
        }
    ]
}
roadslayer = {
    "id": "Roads",
    "maptype": "mapbox", 
    "order": 2,
    "type": "line",
    "source": {
        "type": "geojson",
        "data": roadslines
    },
    "paint": {
        "line-color": "rgba(128,0,128,0.65)",
        "line-width": 6, 
        "line-blur": 2, 
        "line-opacity": 0.75
    },
    "layout": {
        "line-join": "round"        
    }
}

dangerzones = {
    "type": "FeatureCollection", 
    "features": [
        {
            "type": "Feature", 
            "properties": {
                "name": "Danger Zone"
            }, 
            "geometry": {
                "type": "Polygon", 
                "coordinates": [
                    [[-71.08828067779541, 42.360890561289295],
                    [-71.08802318572998, 42.35032997408756],
                    [-71.07295989990234, 42.35591176680853],
                    [-71.07583522796631, 42.3609539828782],
                    [-71.08828067779541, 42.360890561289295]]
                ]
            }
        }
    ]
}
dglayer = {
    "id": "Danger Zone",
    "maptype": "mapbox", 
    "order": 3,
    "type": "fill",
    "source": {
        "type": "geojson",
        "data": dangerzones
    },
    "paint": {
        "fill-antialias": True, 
        "fill-color": "rgba(248,64,0,1.0)",
        "fill-outline-color": "#ff0000"
    },
    "layout": {}
}

custompt = {
    "type": "FeatureCollection", 
    "features": [
        {
            "type": "Feature", 
            "properties": {}, 
            "geometry": {
                "type": "Point", 
                "coordinates": [-71.0771, 42.3599]
            }
        }, 
        {
            "type": "Feature", 
            "properties": {}, 
            "geometry": {
                "type": "Point", 
                "coordinates": [-71.0771, 42.3610]
            }
        }
    ]
}
customLayer = {
    "id": "specialdata",
    "maptype": "mapbox", 
    "order": 1,
    "type": "circle",
    "source": {
        "type": "geojson",
        "data": custompt
    },
    "paint": {
        "circle-color": "rgba(0,0,255,0.85)", 
        "circle-radius": 20
    },
    "layout": {}
}

display(homesdf)









from pixiedust.display.app import *
from pixiedust.utils.shellAccess import ShellAccess
import geojson

@PixieApp
class MapboxUserLayers:
    
    @route()
    def main(self):
        self.USERLAYERS = []
        for key in ShellAccess:
            v = ShellAccess[key]
            if isinstance(v, dict) and "source" in v and "type" in v["source"] and v["source"]["type"] == "geojson" and "id" in v and "paint" in v and "layout" in v and "data" in v["source"]:
#                 gj = geojson.loads(v["source"]["data"])
#                 isvalid = geojson.is_valid(gj)
#                 if isvalid["valid"] == "yes":
                self.USERLAYERS.append(v)
#                 else:
#                     print("Invalid GeoJSON: {0}".format(str(v["source"]["data"])))

        return """<pre>{% for layer in this.USERLAYERS %}
        var layertype = "circle";
        {% if layer["type"] %}
        layertype = "{{layer["type"]}}";
        {%endif%}

        var layerpaint = "{}";
        {% if layer["paint"] %}
        layerpaint = "{{layer["paint"]}}";
        {%endif%}

        var layerlayout = "{}";
        {% if layer["layout"] %}
        layerlayout = "{{layer["layout"]}}";
        {%endif%}

        map.addLayer({
            "id": "{{layer["id"]}}", 
            "type": layertype, 
            "source": {{layer["source"]}},
            "paint": layerpaint, 
            "layout": layerlayout
        });
        {% endfor %}</pre>
"""

mbl = MapboxUserLayers()
mbl.run()

get_ipython().magic('pixiedustLog -l debug')



