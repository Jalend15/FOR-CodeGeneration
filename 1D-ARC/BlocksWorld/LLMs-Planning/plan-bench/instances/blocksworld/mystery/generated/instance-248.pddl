(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects i c h g k l f e a)
(:init 
(harmony)
(planet i)
(planet c)
(planet h)
(planet g)
(planet k)
(planet l)
(planet f)
(planet e)
(planet a)
(province i)
(province c)
(province h)
(province g)
(province k)
(province l)
(province f)
(province e)
(province a)
)
(:goal
(and
(craves i c)
(craves c h)
(craves h g)
(craves g k)
(craves k l)
(craves l f)
(craves f e)
(craves e a)
)))