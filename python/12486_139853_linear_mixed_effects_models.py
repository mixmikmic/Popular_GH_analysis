get_ipython().magic('matplotlib inline')
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from edward.models import Normal

plt.style.use('ggplot')
ed.set_seed(42)

# s - students - 1:2972
# d - instructors - codes that need to be remapped
# dept also needs to be remapped
data = pd.read_csv('../examples/data/insteval.csv')
data['dcodes'] = data['d'].astype('category').cat.codes
data['deptcodes'] = data['dept'].astype('category').cat.codes
data['s'] = data['s'] - 1

train = data.sample(frac=0.8)
test = data.drop(train.index)

train.head()

s_train = train['s'].values.astype(int)
d_train = train['dcodes'].values.astype(int)
dept_train = train['deptcodes'].values.astype(int)
y_train = train['y'].values.astype(float)
service_train = train['service'].values.astype(int)
n_obs_train = train.shape[0]

s_test = test['s'].values.astype(int)
d_test = test['dcodes'].values.astype(int)
dept_test = test['deptcodes'].values.astype(int)
y_test = test['y'].values.astype(float)
service_test = test['service'].values.astype(int)
n_obs_test = test.shape[0]

n_s = 2972  # number of students
n_d = 1128  # number of instructors
n_dept = 14  # number of departments
n_obs = train.shape[0]  # number of observations

# Set up placeholders for the data inputs.
s_ph = tf.placeholder(tf.int32, [None])
d_ph = tf.placeholder(tf.int32, [None])
dept_ph = tf.placeholder(tf.int32, [None])
service_ph = tf.placeholder(tf.float32, [None])

# Set up fixed effects.
mu = tf.Variable(tf.random_normal([]))
service = tf.Variable(tf.random_normal([]))

sigma_s = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([]))))
sigma_d = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([]))))
sigma_dept = tf.sqrt(tf.exp(tf.Variable(tf.random_normal([]))))

# Set up random effects.
eta_s = Normal(loc=tf.zeros(n_s), scale=sigma_s * tf.ones(n_s))
eta_d = Normal(loc=tf.zeros(n_d), scale=sigma_d * tf.ones(n_d))
eta_dept = Normal(loc=tf.zeros(n_dept), scale=sigma_dept * tf.ones(n_dept))

yhat = tf.gather(eta_s, s_ph) +     tf.gather(eta_d, d_ph) +     tf.gather(eta_dept, dept_ph) +     mu + service * service_ph
y = Normal(loc=yhat, scale=tf.ones(n_obs))

q_eta_s = Normal(
    loc=tf.Variable(tf.random_normal([n_s])),
    scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_s]))))
q_eta_d = Normal(
    loc=tf.Variable(tf.random_normal([n_d])),
    scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_d]))))
q_eta_dept = Normal(
    loc=tf.Variable(tf.random_normal([n_dept])),
    scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_dept]))))

latent_vars = {
    eta_s: q_eta_s,
    eta_d: q_eta_d,
    eta_dept: q_eta_dept}
data = {
    y: y_train, 
    s_ph: s_train,
    d_ph: d_train,
    dept_ph: dept_train,
    service_ph: service_train}
inference = ed.KLqp(latent_vars, data)

yhat_test = ed.copy(yhat, {
    eta_s: q_eta_s.mean(),
    eta_d: q_eta_d.mean(),
    eta_dept: q_eta_dept.mean()})

inference.initialize(n_print=2000, n_iter=10000)
tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
  # Update and print progress of algorithm.
  info_dict = inference.update()
  inference.print_progress(info_dict)

  t = info_dict['t']
  if t == 1 or t % inference.n_print == 0:
    # Make predictions on test data.
    yhat_vals = yhat_test.eval(feed_dict={
        s_ph: s_test,
        d_ph: d_test,
        dept_ph: dept_test,
        service_ph: service_test})

    # Form residual plot.
    plt.title("Residuals for Predicted Ratings on Test Set")
    plt.xlim(-4, 4)
    plt.ylim(0, 800)
    plt.hist(yhat_vals - y_test, 75)
    plt.show()

student_effects_lme4 = pd.read_csv('../examples/data/insteval_student_ranefs_r.csv')
instructor_effects_lme4 = pd.read_csv('../examples/data/insteval_instructor_ranefs_r.csv')
dept_effects_lme4 = pd.read_csv('../examples/data/insteval_dept_ranefs_r.csv')

student_effects_edward = q_eta_s.mean().eval()
instructor_effects_edward = q_eta_d.mean().eval()
dept_effects_edward = q_eta_dept.mean().eval()

plt.title("Student Effects Comparison")
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("Student Effects from lme4")
plt.ylabel("Student Effects from edward")
plt.scatter(student_effects_lme4["(Intercept)"], 
            student_effects_edward, 
            alpha = 0.25)
plt.show()

plt.title("Instructor Effects Comparison")
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xlabel("Instructor Effects from lme4")
plt.ylabel("Instructor Effects from edward")
plt.scatter(instructor_effects_lme4["(Intercept)"], 
            instructor_effects_edward, 
            alpha = 0.25)
plt.show()

#  Add in the intercept from R and edward
dept_effects_and_intercept_lme4 = 3.28259 + dept_effects_lme4["(Intercept)"]
dept_effects_and_intercept_edward = mu.eval() + dept_effects_edward

plt.title("Departmental Effects Comparison")
plt.xlim(3.0, 3.5)
plt.ylim(3.0, 3.5)
plt.xlabel("Department Effects from lme4")
plt.ylabel("Department Effects from edward")
plt.scatter(dept_effects_and_intercept_lme4, 
            dept_effects_and_intercept_edward,
            s = 0.01*train.dept.value_counts())
plt.show()



