(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects a b h d f)
(:init 
(harmony)
(planet a)
(planet b)
(planet h)
(planet d)
(planet f)
(province a)
(province b)
(province h)
(province d)
(province f)
)
(:goal
(and
(craves a b)
(craves b h)
(craves h d)
(craves d f)
)))