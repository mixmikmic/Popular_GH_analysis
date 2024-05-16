from planning import *

get_ipython().magic('psource Action')

get_ipython().magic('psource PDDL')

from utils import *
# this imports the required expr so we can create our knowledge base

knowledge_base = [
    expr("Connected(Bucharest,Pitesti)"),
    expr("Connected(Pitesti,Rimnicu)"),
    expr("Connected(Rimnicu,Sibiu)"),
    expr("Connected(Sibiu,Fagaras)"),
    expr("Connected(Fagaras,Bucharest)"),
    expr("Connected(Pitesti,Craiova)"),
    expr("Connected(Craiova,Rimnicu)")
    ]

knowledge_base.extend([
     expr("Connected(x,y) ==> Connected(y,x)"),
     expr("Connected(x,y) & Connected(y,z) ==> Connected(x,z)"),
     expr("At(Sibiu)")
    ])

knowledge_base

#Sibiu to Bucharest
precond_pos = [expr('At(Sibiu)')]
precond_neg = []
effect_add = [expr('At(Bucharest)')]
effect_rem = [expr('At(Sibiu)')]
fly_s_b = Action(expr('Fly(Sibiu, Bucharest)'), [precond_pos, precond_neg], [effect_add, effect_rem])

#Bucharest to Sibiu
precond_pos = [expr('At(Bucharest)')]
precond_neg = []
effect_add = [expr('At(Sibiu)')]
effect_rem = [expr('At(Bucharest)')]
fly_b_s = Action(expr('Fly(Bucharest, Sibiu)'), [precond_pos, precond_neg], [effect_add, effect_rem])

#Sibiu to Craiova
precond_pos = [expr('At(Sibiu)')]
precond_neg = []
effect_add = [expr('At(Craiova)')]
effect_rem = [expr('At(Sibiu)')]
fly_s_c = Action(expr('Fly(Sibiu, Craiova)'), [precond_pos, precond_neg], [effect_add, effect_rem])

#Craiova to Sibiu
precond_pos = [expr('At(Craiova)')]
precond_neg = []
effect_add = [expr('At(Sibiu)')]
effect_rem = [expr('At(Craiova)')]
fly_c_s = Action(expr('Fly(Craiova, Sibiu)'), [precond_pos, precond_neg], [effect_add, effect_rem])

#Bucharest to Craiova
precond_pos = [expr('At(Bucharest)')]
precond_neg = []
effect_add = [expr('At(Craiova)')]
effect_rem = [expr('At(Bucharest)')]
fly_b_c = Action(expr('Fly(Bucharest, Craiova)'), [precond_pos, precond_neg], [effect_add, effect_rem])

#Craiova to Bucharest
precond_pos = [expr('At(Craiova)')]
precond_neg = []
effect_add = [expr('At(Bucharest)')]
effect_rem = [expr('At(Craiova)')]
fly_c_b = Action(expr('Fly(Craiova, Bucharest)'), [precond_pos, precond_neg], [effect_add, effect_rem])

#Drive
precond_pos = [expr('At(x)')]
precond_neg = []
effect_add = [expr('At(y)')]
effect_rem = [expr('At(x)')]
drive = Action(expr('Drive(x, y)'), [precond_pos, precond_neg], [effect_add, effect_rem])

def goal_test(kb):
    return kb.ask(expr("At(Bucharest)"))

prob = PDDL(knowledge_base, [fly_s_b, fly_b_s, fly_s_c, fly_c_s, fly_b_c, fly_c_b, drive], goal_test)





