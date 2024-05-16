# Import the SWAT package which contains the CAS interface
import swat

# Create a CAS session on mycas1 port 12345
conn = swat.CAS('mycas1', 12345, 'username', 'password') 

# Run the builtins.listnodes action
nodes = conn.listnodes()
nodes

# Grab the nodelist DataFrame
df = nodes['nodelist']
df

roles = df[['name', 'role']]
roles

# Extract the worker nodes using a DataFrame mask
roles[roles.role == 'worker']

# Extract the controllers using a DataFrame mask
roles[roles.role == 'controller']

conn.close()



