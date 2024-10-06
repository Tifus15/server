import sympy as sym

a,b,db = sym.symbols("l th dth")

l = [1,1.5, 1]
m = [1,3,2]
th = sym.symbols("th(1:4)")
g = 10

pth = sym.symbols("pth(1:4)")
dth = sym.symbols("dth(1:4)")

xi = a*sym.sin(b)
yi = - a*sym.cos(b)

dxi =db*a*sym.cos(b)
dyi =db*a*sym.sin(b)

x=[]
y=[]
dx=[]
dy=[]

vs = []

x.append(xi.subs([(a,l[0]),(b,th[0])]))
y.append(yi.subs([(a,l[0]),(b,th[0])]))

x.append(x[0] + xi.subs([(a,l[1]),(b,th[1])]))
y.append(y[0] + yi.subs([(a,l[1]),(b,th[1])]))

x.append(x[1] + xi.subs([(a,l[2]),(b,th[2])]))
y.append(y[1] + yi.subs([(a,l[2]),(b,th[2])]))

dx.append(dxi.subs([(a,l[0]),(b,th[0]),(db,dth[0])]))
dy.append(dyi.subs([(a,l[0]),(b,th[0]),(db,dth[0])]))

dx.append(dx[0] + dxi.subs([(a,l[1]),(b,th[1]),(db,dth[1])]))
dy.append(dy[0] + dyi.subs([(a,l[1]),(b,th[1]),(db,dth[1])]))

dx.append(dx[1] + dxi.subs([(a,l[2]),(b,th[2]),(db,dth[2])]))
dy.append(dy[1] + dyi.subs([(a,l[2]),(b,th[2]),(db,dth[2])]))

vs.append(sym.simplify(dx[0]**2 + dy[0]**2))
vs.append(sym.simplify(dx[1]**2 + dy[1]**2))
vs.append(sym.simplify(dx[2]**2 + dy[2]**2))

K = m[0]*vs[0]/2 + m[1]*vs[1]/2 +m[2]*vs[2]/2
U = m[0]*g*y[0] + m[1]*g*y[1] +m[2]*g*y[2]

L = K-U
#print(L)

p_theta1 = sym.simplify(sym.diff(L,dth[0]))
p_theta2 = sym.simplify(sym.diff(L,dth[1]))
p_theta3 = sym.simplify(sym.diff(L,dth[2]))
p_theta = sym.Matrix([[p_theta1],[p_theta2],[p_theta3]])
pth_vec=sym.Matrix([[pth[0]],[pth[1]],[pth[2]]])
Y = sym.Matrix([[dth[0]],[dth[1]],[dth[2]]])
M = p_theta.jacobian(Y)

M_inv = sym.simplify(M**-1)
"""
d_th = M_inv @ pth_vec
#print(d_th)

H = K + U

H = sym.simplify(H.subs([(dth[0],d_th[0]),(dth[1],d_th[1]),(dth[2],d_th[2])]))

print(H)
"""