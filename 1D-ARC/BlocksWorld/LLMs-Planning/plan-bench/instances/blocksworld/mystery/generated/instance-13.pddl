(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects i d a h c e j l f g k)
(:init 
(harmony)
(planet i)
(planet d)
(planet a)
(planet h)
(planet c)
(planet e)
(planet j)
(planet l)
(planet f)
(planet g)
(planet k)
(province i)
(province d)
(province a)
(province h)
(province c)
(province e)
(province j)
(province l)
(province f)
(province g)
(province k)
)
(:goal
(and
(craves i d)
(craves d a)
(craves a h)
(craves h c)
(craves c e)
(craves e j)
(craves j l)
(craves l f)
(craves f g)
(craves g k)
)))