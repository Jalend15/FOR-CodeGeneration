(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects k j a c l h i)
(:init 
(harmony)
(planet k)
(planet j)
(planet a)
(planet c)
(planet l)
(planet h)
(planet i)
(province k)
(province j)
(province a)
(province c)
(province l)
(province h)
(province i)
)
(:goal
(and
(craves k j)
(craves j a)
(craves a c)
(craves c l)
(craves l h)
(craves h i)
)))