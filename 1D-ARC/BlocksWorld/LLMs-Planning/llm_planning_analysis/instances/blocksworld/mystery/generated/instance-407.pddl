(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects e c h l k d)
(:init 
(harmony)
(planet e)
(planet c)
(planet h)
(planet l)
(planet k)
(planet d)
(province e)
(province c)
(province h)
(province l)
(province k)
(province d)
)
(:goal
(and
(craves e c)
(craves c h)
(craves h l)
(craves l k)
(craves k d)
)))