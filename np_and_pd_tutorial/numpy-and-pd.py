# %%


# %%


# %%


# %%


# %%
#

# %%
import numpy as np
# import sys
# import time
a=np.array([[1,2],[3,4],[5,6]],dtype=np.float64)
# print(a)
# print(a.ndim)
# print(a.itemsize)
# print(a.shape)
'''a=np.array([[1,2],[3,4],[5,6]],dtype=np.complex64)
a=np.array(["gggg"]+['jjj'])
# print(a)
a=np.zeros(3)
a=np.arange(5)
a=np.char.add(['hello'],['john'])
a=np.char.multiply(['hello'],3)
a=np.char.center(['hello'],20,fillchar='-')
a=np.char.capitalize('hello')
a=np.char.title('HELLO MY FRIEND')

a=np.char.lower('HELLO MY FRIEND')
a=np.char.upper('hello my friend')
a=np.char.split('hello my friend')
a=np.char.splitlines('hel\nlo m\ny friend')
a=np.char.strip(['helloa','mya','frienda'],'a')
a=np.char.join('-','mdy')
a=np.char.replace('he is a good dance and is stil','is','was')

# ARRAY MANIPULATION
# ARRAY MANIPULATION-changing shape
a=np.arange(9)
print ('ORIGINAL ARRAY')
print(a)
b=a.reshape(3,3)
print('modified array\n',b)
print(b.flatten())

a=np.arange(12).reshape(4,3)
# print(a)
b=np.transpose(a)
b=np.arange(8).reshape(2,4)
b=np.arange(8).reshape(2,4)
print(b)
c= b.reshape(2,2,2)
print(c)
d=np.rollaxis(c,1)
print('\nrollaxis \n',d)
d=np.rollaxis(c,2)
print('\nrollaxis \n',d)

b=np.arange(8).reshape(2,4)

c= b.reshape(2,2,2)
c=np.swapaxes(c,1,2)
print(c)

# NUMPY ARITHMETIC
a=np.arange(9).reshape(3,3)
print(a)
print()
b=np.array([10,11,12])
print(b)
print()
c=np.add(a,b)
c=np.subtract(a,b)
c=np.multiply(a,b)
c=np.divide(a,b)

print(c)

# SLICING
a=np.arange(20)
print(a[4:])
s=slice(2,9,2)
print(a[s])

# ITERATING OVER ARRAY
a=np.arange(0,45,5)
a=a.reshape(3,3)
print(a)
# for x in np.nditer(a):
#     print(x)

# ITERATOR ORDER [C-STYLE AND F-STYLE]
for x in np.nditer(a,order='C'):
    print(x)
for x in np.nditer(a,order='F'):
    print(x)

# JOINING ARRAYS
a=np.array([[1,2],[3,4]])
print("first arraye:\n",a)
b=np.array([[5,6],[7,8]])
print("second arraye:\n",b)
print('\n')

print(np.concatenate((a,b)))
print('\n')
print(np.concatenate((a,b),axis=1))



# splitting array

a=np.arange(9)
print(a)
print(np.split(a,3))

# %%
from matplotlib import pyplot as plt

a=np.array([20,87,4,40,53,74,56,51,11,20,40,15,79,25,27]
)
plt.hist(a,bins=[0,20,40,60,80,100])
plt.title("histogram")
plt.show()


# %%
#Splitting arrays

# %%
a= np.arange(9)
a

# %%
np.split(a,3)


# %%
np.split(a,[4,5])

# %%
np.split(a,[4,7])

# %% [markdown]
# resizing the array
# 

# %%
a=np.array([[1,2,3],[4,5,6]])
print(a)
a.shape
print('\n')
b=np.resize(a,(3,3))
b

# %% [markdown]
# histogram
# 

# %%
a=np.array([20,87,4,40,53,74,56,51,11,20,40,15,79,25,27]
)
plt.hist(a,bins=[0,20,40,60,80,100])
plt.title("histogram")
plt.show()

# %% [markdown]
# linspace function
# 

# %%
a=np.linspace(1,3,10)
a

# %%
# sum and axis
a=np.array([[1,2,3],[3,4,5]])
a.sum(axis=0)

# %% [markdown]
# Square root and standard deviation
# 

# %%
a=np.array([[1,2,3],[3,4,5]])
print(np.sqrt(a),'\n',
np.std(a))

# %% [markdown]
# Ravel function

# %%
a=np.array([[1,2,3],[3,4,5]])
a.ravel()

# %%
a=np.array([[1,2,3],[3,4,5]])
np.log10(a)

# %% [markdown]
# PRACTICE EXAMPLES

# %%
x=np.arange(0,3*np.pi,0.1)
y=np.sin(x)
plt.plot(x,y)
plt.show()

# %%
Z=np.zeros((6,6),dtype=int)
Z[1::2,::2]=1
Z[::2,1::2]=1
Z

# %%
z=np.random.rand(10,10)
z[np.random.randint(10,size=5),np.random.randint(10,size=5)]=np.nan
z


# %%
print ("Total missing num :",np.isnan(z).sum())
print("Indexes of missing value \n",np.argwhere(np.isnan(z)))
inds=np.where(np.isnan(z))
print(inds)
z[inds]=0
print(z)'''



# %%
# PANDAS TUTORIAL
import pandas as pd


# SERIES CREATE,MANIPULATE,QUERY,DELETE

# creating a series from a list

a=[0,1,2,3,4]
# print(pd.__version__)
s1=pd.Series(a)
# print(s1)


# order=[1,2,3,4,5]
# s2=pd.Series(a,index=order)
# print(s2)

n=np.random.randn(5)
index=['a','b','c','d','e']
s2=pd.Series(n,index=index)

d={'a':1,'b':3.6,'c':2}
s2=pd.Series(d)

# /MODIFYING THE INDEX
s1.index=['A','B','C','D','F']
# print(s1)

# SLICING
a=s1[:3]
s3=s1._append(s2)
s3=s3.drop('c')
print(s3)