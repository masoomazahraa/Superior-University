def isgoal(state,goal):
    return state[0]==goal or state[1]==goal
def getsuccessors(state,jug1cap,jug2cap):
    successors=[]
    jug1,jug2=state
    successors.append((jug1cap,jug2))
    successors.append((jug1,jug2cap))
    successors.append((0,jug2))
    successors.append((jug1, 0))
    intojug2=min(jug1,jug2cap-jug2)
    successors.append((jug1-intojug2,jug2+intojug2))
    intojug1= min(jug2,jug1cap-jug1)
    successors.append((jug1 +intojug1, jug2-intojug1))
    return successors
def dfs(jug1cap,jug2cap,goal):
    stack=[(0,0)]
    visited=set()
    path=[]
    while stack:
        current = stack.pop()
        path.append(current)
        if isgoal(current,goal):
            return path
        if current in visited:
            path.pop()
            continue
        visited.add(current)
        for successor in getsuccessors(current,jug1cap, jug2cap):
            if successor not in visited:
                stack.append(successor)
    return None
def printsol(solution):
    if solution:
        for i,state in enumerate(solution):
            print(f"Step{i}: Jug1= {state[0]} liters,Jug2= {state[1]} liters")
    else:
        print("No solution found:')")
jug1cap=4
jug2cap=3
goal=2
solution=dfs(jug1cap,jug2cap,goal)
printsol(solution)