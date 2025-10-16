#!/usr/bin/env python
# coding: utf-8

# # 5. 이산형 확률변수

# ## 5.1. 1차원 이산형 확률변수

# ### 5.1.1. 1차원 이산형 확률변수의 정의

# In[8]:


import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('precision', '3')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


# 불공정한 주사위의 확률분포
x_set = np.array([1, 2, 3, 4, 5, 6])


# In[10]:


# 확률변수 구현 (PMF)
def f(x):
    if x in x_set:
        return x / 21
    else:
        return 0


# In[11]:


X = [x_set, f]


# In[12]:


# 확률 p_k를 구한다
prob = np.array([f(x_k) for x_k in x_set])
# x_k와 p_k의 대응을 사전식으로 표시
dict(zip(x_set, prob))


# In[13]:


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.bar(x_set, prob)
ax. set_xlabel('value')
ax. set_ylabel('probability')

plt.show()


# #### **참고: 딕셔너리

# In[15]:


a = {'사과':1, '딸기':5, '귤':10}


# In[16]:


a


# In[17]:


a = {('초콜릿', 200):20, ('마카롱', 500):15, ('쿠키', 300):30}
a


# In[18]:


a = {'사과':1, '딸기':5, '귤':10}
v1 = a['딸기']
v1


# In[33]:


v2 = a['레몬']  # 존재하지 않기 때문에 오류 발생
v2  


# In[34]:


f1 = '딸기' in a
f1


# In[35]:


f2 = '레몬' not in a
f2


# In[36]:


f3 = '레몬' in a
f3


# In[37]:


v1 = a.get('딸기')
v1


# In[38]:


v2 = a.get('레몬')
v2


# In[39]:


a = {'초콜릿':1, '마카롱':2, '쿠키':3}
a['초콜릿'] = 'One'
a['마카롱'] = 'Two'
a['쿠키'] = 'Three'
a


# In[40]:


d = dict(초콜릿 = 20, 마카롱 = 15, 쿠키 = 30)
d


# In[45]:


key = ['초콜릿', '마카롱', '쿠키']
value = [20, 15, 30]
d = dict(zip(key, value))  #zip으로 묶음
d


# In[46]:


d = dict([('초콜릿', 20), ('마카롱', 15), ('쿠키', 30)])
d


# **

# In[47]:


# 확률의 성질
np.all(prob >= 0) # 확률은 절대적으로 0이상


# In[48]:


np.sum(prob) # 모든 확률의 합은 1


# In[49]:


# 누적분포함수
def F(x):
    return np.sum([f(x_k) for x_k in x_set if x_k <= x])


# In[50]:


F(3)


# In[51]:


# 확률변수의 변환
y_set = np.array([2 * x_k + 3 for x_k in x_set])
prob = np.array([f(x_k) for x_k in x_set])
dict(zip(y_set, prob))


# ### 5.1.2. 1차원 이산형 확률변수의 지표

# In[52]:


# 기댓값(확률변수의 평균)
np.sum([x_k * f(x_k) for x_k in x_set])


# #### **참고: 데이터 샘플링

# In[53]:


np.random.choice(5, 5, replace=False) # shuffle 명령과 같다.


# In[54]:


np.random.choice(5, 3, replace=False) # 3개만 선택


# In[55]:


np.random.choice(5, 10) # 반복해서 10개 선택


# In[56]:


np.random.choice(5, 10, p=[0.1, 0, 0.3, 0.6, 0]) # 선택 확률을 다르게해서 10개 선택


# **

# In[57]:


sample = np.random.choice(x_set, int(1e6), p=prob)
np.mean(sample)


# In[58]:


def E(X, g=lambda x: x):
    x_set, f = X
    return np.sum([g(x_k) * f(x_k) for x_k in x_set])


# In[59]:


E(X)


# In[60]:


E(X, g=lambda x: 2*x + 3)


# In[61]:


2 * E(X) + 3


# #### **참고: 람다(lambda) 함수(익명 함수)

# In[68]:


strings = ['hyeja', 'parkhyeja', 'youngtae', 'kimyoungtae', 'bbangtae']


# In[69]:


strings.sort(key=lambda x: len(set(list(x))))


# In[70]:


strings


# **

# In[71]:


# 분산
mean = E(X)
np.sum([(x_k-mean)**2 * f(x_k) for x_k in x_set])


# In[72]:


def V(X, g=lambda x: x):
    x_set, f = X
    mean = E(X, g)
    return np.sum([(g(x_k)-mean)**2 * f(x_k) for x_k in x_set])


# In[73]:


V(X)


# In[74]:


V(X, lambda x: 2*x + 3)


# In[75]:


2**2 * V(X)


# In[ ]:




