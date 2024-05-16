from pixiedust.display.app import *
from pixiedust.display.streaming.data import *
from pixiedust.display.streaming.bokeh import *
import requests

@PixieApp
class MessageHubStreamingApp():
    def setup(self):
        self.streamingDisplay = None
        self.streamingData = None
        self.contents = []
        self.schemaX = None      

    def newDisplayHandler(self, options, entity):
        if self.streamingDisplay is None:
            self.streamingDisplay = LineChartStreamingDisplay(options, entity)
        return self.streamingDisplay
    
    def getTopics(self):
        rest_endpoint = "https://kafka-rest-prod01.messagehub.services.us-south.bluemix.net:443"
        headers = {
            'X-Auth-Token': self.credentials["api_key"],
            'Content-Type': 'application/json'
        }        
        return requests.get('{}/topics'.format(rest_endpoint),headers = headers).json()
    
    @route()
    def mainScreen(self):
        return """
<div class="well" style="text-align:center">
    <div style="font-size:x-large">MessageHub Streaming Browser.</div>
    <div style="font-size:large">Click on a topic to start</div>
</div>

{%for topic in this.getTopics()%}
    {%if loop.first or ((loop.index % 4) == 1)%}
<div class="row">
    <div class="col-sm-2"/>
    {%endif%}
    <div pd_options="topic=$val(topic{{loop.index}}{{prefix}})" class="col-sm-2" style="border: 1px solid lightblue;margin: 10px;border-radius: 25px;cursor:pointer;
        min-height: 150px;background-color:#e6eeff;display: flex;align-items: center;justify-content:center">
        <span id="topic{{loop.index}}{{prefix}}">{{topic}}</span>
    </div>
    {%if loop.last or ((loop.index % 4) == 0)%}
    <div class="col-sm-2"/>
</div>
    {%endif%}
{%endfor%}
        """
    
    def displayNextTopics(self):
        payload = self.streamingData.getNextData()
        if payload is not None and len(payload)>0:
            self.contents = self.contents + payload
            self.contents = self.contents[-10:]                
            html = ""
            for event in self.contents:
                html += "{}<br/>".format(json.dumps(event))
            print(html)
            
    def newDisplayHandler(self, options, entity):
        if self.streamingDisplay is None:
            self.streamingDisplay = LineChartStreamingDisplay(options, entity)
        return self.streamingDisplay
            
    @route(topic="*",streampreview="*",schemaX="*")
    def showChart(self, schemaX):
        self.schemaX = schemaX
        self.avgChannelData = self.streamingData.getStreamingChannel(self.computeAverages)
        return """
<div class="well" style="text-align:center">
    <div style="font-size:x-large">Real-time chart for {{this.schemaX}}(average).</div>
</div>
<style>
.bk-root{
display:flex;
justify-content:center;
}
</style>
<div pd_refresh_rate="1000" pd_entity="avgChannelData"></div>
        """
    
    def computeAverages(self, avg, newData):
        newValue = []
        for jsonValue in newData:
            if self.schemaX in jsonValue:
                thisValue = float(jsonValue[self.schemaX])
                avg = thisValue if avg is None else (avg + thisValue)/2
                newValue.append(avg)
        return newValue, avg
        
    
    @route(topic="*",streampreview="*")
    def createStreamWidget(self, streampreview):
        if streampreview=="realtimeChart":
            return """
<div>
    {%for key in this.streamingData.schema%}
    {%if loop.first%}
    <div class="well" style="text-align:center">
        <div style="font-size:x-large">Create a real-time chart by selecting a field.</div>
    </div>
    {%endif%}
    <div class="radio" style="margin-left:20%">
      <label>
          <input type="radio" pd_options="streampreview=""" + streampreview + """;schemaX=$val(schemaX{{loop.index}}{{prefix}})" 
              id="schemaX{{loop.index}}{{prefix}}" pd_target="realtimeChartStreaming{{prefix}}" 
              name="schemaX" value="{{key}}">{{key}}
      </label>
    </div>
    {%endfor%}
</div>"""
        return """<div pd_refresh_rate="1000" pd_script="self.displayNextTopics()"></div>"""
        
    @route(topic="*")
    def previewTopic(self, topic):
        self.topic = topic
        if self.streamingData is not None:
            self.streamingData.close()
        self.streamingData = MessagehubStreamingAdapter( self.topic, self.credentials["username"], self.credentials["password"] )
        return """
<div class="row">
    <div class="col-sm-12" id="targetstreaming{{prefix}}">
        <div pd_refresh_rate="1000" style="white-space:nowrap;overflow-x:auto;border:aliceblue 2px solid;height:17em;line-height:1.5em">
            <pd_script>self.displayNextTopics()</pd_script>
            <div style="width:100px;height:60px;left:47%;position:relative">
                <i class="fa fa-circle-o-notch fa-spin" style="font-size:48px"></i>
            </div>
            <div style="text-align:center">Waiting for data from MessageHub</div>
        </div>
    </div>
</div>
<div class="row" id="realtimeChartStreaming{{prefix}}">
    <div pd_refresh_rate="4000" pd_options="streampreview=realtimeChart">
    </div>
</div>
        """
        
    
MessageHubStreamingApp().run(credentials={
    "username": "XXXX",
    "password": "XXXX",
    "api_key" : "XXXX"
})



