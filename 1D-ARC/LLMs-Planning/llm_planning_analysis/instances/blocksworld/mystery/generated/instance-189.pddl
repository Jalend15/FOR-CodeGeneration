(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects k g c i b)
(:init 
(harmony)
(planet k)
(planet g)
(planet c)
(planet i)
(planet b)
(province k)
(province g)
(province c)
(province i)
(province b)
)
(:goal
(and
(craves k g)
(craves g c)
(craves c i)
(craves i b)
)))