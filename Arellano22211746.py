"""
Práctica: Sistema cardiovascular

Departamento de Ingeniería Eléctrica y Electrónica, Ingeniería Biomédica
Tecnológico Nacional de México [TecNM - Tijuana]
Blvd. Alberto Limón Padilla s/n, C.P. 22454, Tijuana, B.C., México

Nombre de la alumna: Bricia Idalia Arellano Salones
Número de control: 22211746
Correo institucional: L22211746@tectijuana.edu.mx

Asignatura: Modelado de Sistemas Fisiológicos
Docente: Dr. Paul Antonio Valle Trujillo; paul.valle@tectijuana.edu.mx
"""
# Instalar librerias en consola
#!pip install control
#!pip install slycot

# Librerías para cálculo numérico y generación de gráficas
import numpy as np
import math as m
import matplotlib.pyplot as plt
import control as ctrl
from scipy import signal
import pandas as pd

# u = np.array(pd.read_excel('signal.xlsx', header = None))
x0, t0, tend, dt, w, h = 0,0,10, 1E-3, 10, 5
n = round((tend - t0)/dt) + 1
t = np.linspace(t0, tend, n)
u = np.zeros(n); u[round(1/dt):round(2/dt)] = 1

def musculo(a, Cs, Cp, R):
    num = [Cs*R, 1 - a]
    den = [R*(Cs+Cp), 1]
    sys = ctrl.tf(num,den)
    return sys

#Función de transferencia: Control
a, Cs, Cp, R = 0.25, 10E-6, 100E-6, 100
sysctrl = musculo(a, Cs, Cp, R)
print(f'Función de transferencia del control: {sysctrl}')

#Función de transferencia: Caso
a, Cs, Cp, R = 0.25, 10E-6, 100E-6, 10E3
syscaso = musculo(a, Cs, Cp, R)
print(f'Función de transferencia del caso: {syscaso}')

_, Fs1 = ctrl.forced_response(sysctrl, t,u,x0)
_, Fs2 = ctrl.forced_response(syscaso, t,u,x0)

clr1 = np.array([0.4980, 0.2980, 0.6471])
clr2 = np.array([0.0196, 0.4980, 0.4275])
clr3 = np.array([0.8745, 0.2235, 0.4118])
clr4 = np.array([0.2000, 0.7804, 0.8471])
clr5 = np.array([1.0000, 0.5529, 0.0784])
clr6 = np.array([0.4824, 0.1765, 0.2627])

fg1 = plt.figure()
plt.plot(t, u,'-', color = clr1, label ='Ve(t)')
plt.plot(t,Fs1, '-', linewidth=1, color= clr2, label='Fs(t): Control' )
plt.plot(t,Fs2, '-', linewidth=1, color= clr3, label='Fs(t): Caso' )

plt.grid(False)
plt.xlim(0,10); plt.xticks(np.arange(0,11,1))
plt.ylim(-0.1,1.2); plt.yticks(np.arange(-0.1,1.2, 0.2))
plt.xlabel('t [s]')
plt.ylabel('Fi(t) [V]')
plt.legend(bbox_to_anchor=(0.5,-0.2), loc='center',ncol=3)
plt.show()
fg1.set_size_inches(w,h)
fg1.tight_layout()
fg1.savefig('Sistema musculoesqueletico python.png', dpi = 600, bbox_inches = 'tight')
fg1.savefig('Sistema musculoesqueletico python.pdf')

#Controlador PI
def controlador(kP, kI, sys):
    Cr = 1E-6 
    Re = 1/(kI*Cr)
    Rr = kP*Re
    numPI = [Rr*Cr,1]
    denPI = [Re*Cr,0]
    PI = ctrl.tf(numPI,denPI)
    X = ctrl.series(PI, sys)
    sysPI = ctrl.feedback(X,1, sign = -1)
    return sysPI

trtPI = controlador(0.0209824064736628,43250.2057000142,syscaso)

#Respuestas en lazo cerrado
_, Fs3 = ctrl.forced_response(trtPI, t, Fs1, x0)

fg2 = plt.figure()
plt.plot(t, u,'-', color = clr1, label ='Ve(t)')
plt.plot(t,Fs1, '-', linewidth=1, color= clr2, label='Fs(t): Control' )
plt.plot(t,Fs2, '-', linewidth=1, color= clr3, label='Fs(t): Caso' )
plt.plot(t,Fs3, ':', linewidth=1.5, color= clr4, label='Fs(t): Tratamiento' )
plt.grid(False)
plt.xlim(0,10); plt.xticks(np.arange(0,11,1))
plt.ylim(-0.1,1.2); plt.yticks(np.arange(-0.1,1.2, 0.2))
plt.xlabel('Fs(t) [V]')
plt.ylabel('t [s]')
plt.legend(bbox_to_anchor=(0.5,-0.2), loc='center',ncol=3)
plt.show()
fg2.set_size_inches(w,h)
fg2.tight_layout()
fg2.savefig('Sistema musculoesqueletico python hipo PI.png', dpi = 600, bbox_inches = 'tight')
fg2.savefig('Sistema musculoesqueletico python hipo PI.pdf')

