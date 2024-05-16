from IPython.display import HTML

js="""
alert("Hello Javascript (created in python string")
// Lots of pre-written stuff could go here - all generated from Python
"""

# This is Python, printing out the javascript into the browser window
HTML('<script type="text/Javascript">%s</script>' % (js,))

# Nothing will appear be 'output' - but an annoying pop-up will...

html="""
<input type="text" id="textinput" value="12"/>
<input type="submit" id="textsubmit">
"""

# This is Python, printing out the javascript into the browser window
HTML(html)

def recalculate_cell_in_python(v):
    if v % 2 == 0: return v/2
    return v*3+1

# Lots more Python could go here
# You can also have side-effects, etc

# This python import will be 'visible' for the python code executed by the javascript callback
# because that happens 'afterwards' as far as the Python kernel is concerned
import json 

js="""
var kernel = IPython.notebook.kernel;
$('#textsubmit').off('click').on('click', function(e) {

    var javascript_cell_value = $('#textinput').val();

    var cmd=[
      'python_new_value = recalculate_cell_in_python('+javascript_cell_value+')',
      'json.dumps( dict( v=python_new_value ) )'
    ].join(';');

    kernel.execute(cmd, {iopub: {output: handle_python_output}}, {silent:false});

    function handle_python_output(msg) {
      //console.log(msg);
      if( msg.msg_type == "error" ) {
        console.log("Javascript received Python error : ", msg.content);
      }
      else {  // execute_result
        var res_str = msg.content.data["text/plain"];
        // Take off surrounding quotes
        var res=JSON.parse( res_str.replace(/^['"](.*)['"]$/, "$1") ); 
        $('#textinput').val( res.v );
      }
    }
    
    return false;
});
"""

# Again,this is a Python cell, printing out the javascript into the browser window
HTML('<script type="text/Javascript">%s</script>' % (js,))



