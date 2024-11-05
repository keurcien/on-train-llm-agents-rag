import time
from langchain_core.runnables import RunnableLambda

def foo(input_):
    return input_.upper()

def bar(input_):
    return input_ + " hohoho"

foo_runnable = RunnableLambda(foo)
bar_runnable = RunnableLambda(bar)

chain = foo_runnable | bar_runnable

for i in chain.stream("ho"):
    time.sleep(3)
    print(i)