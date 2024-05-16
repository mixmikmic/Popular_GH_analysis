from pixiedust.display.app import *
from pixiedust.display.streaming import *
from pixiedust.display.streaming.bokeh import *

N = 100
class RandomStreamingAdapter(StreamingDataAdapter):
    def __init__(self):
        self.x = np.random.random(size=N) * 100
        self.y = np.random.random(size=N) * 100
    
    def doGetNextData(self):
        rand = 2 * (np.random.random(size=N) - 0.5)
        d = np.sqrt((self.x-50)**2 + (self.y-50)**2)/100
        self.x = self.x + 2 * np.sin(d) * rand
        self.y = self.y + np.cos(d**2) * rand
        #return (self.x, self.y)
        return self.y

@PixieApp
class StreamingApp():    
    def setup(self):
        self.streamingData = RandomStreamingAdapter()
        self.scatter = False
        self.streamingDisplay = None

    def newDisplayHandler(self, options, entity):
        if self.streamingDisplay is None:
            self.streamingDisplay = ScatterPlotStreamingDisplay(options, entity) if self.scatter else LineChartStreamingDisplay(options, entity)
        return self.streamingDisplay
        
    @route()
    def main(self):
        return"""
<button type="button" class="btn btn-default">Toggle
    <pd_script>
self.scatter = not self.scatter
self.streamingDisplay = None
</pd_script>
</button>
            <div pd_entity="streamingData" pd_refresh_rate="1000">
            </div>
        """

#run the app
a = StreamingApp()
a.run(runInDialog='false')

