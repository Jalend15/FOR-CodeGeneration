(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects l c b i d e a k f j h g)
(:init 
(harmony)
(planet l)
(planet c)
(planet b)
(planet i)
(planet d)
(planet e)
(planet a)
(planet k)
(planet f)
(planet j)
(planet h)
(planet g)
(province l)
(province c)
(province b)
(province i)
(province d)
(province e)
(province a)
(province k)
(province f)
(province j)
(province h)
(province g)
)
(:goal
(and
(craves l c)
(craves c b)
(craves b i)
(craves i d)
(craves d e)
(craves e a)
(craves a k)
(craves k f)
(craves f j)
(craves j h)
(craves h g)
)))