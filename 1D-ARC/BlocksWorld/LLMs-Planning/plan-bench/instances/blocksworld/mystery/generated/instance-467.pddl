(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects g i l j a f h)
(:init 
(harmony)
(planet g)
(planet i)
(planet l)
(planet j)
(planet a)
(planet f)
(planet h)
(province g)
(province i)
(province l)
(province j)
(province a)
(province f)
(province h)
)
(:goal
(and
(craves g i)
(craves i l)
(craves l j)
(craves j a)
(craves a f)
(craves f h)
)))