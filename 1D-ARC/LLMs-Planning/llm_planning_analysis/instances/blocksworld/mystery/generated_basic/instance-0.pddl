

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a c)
(craves b a)
(craves c d)
(planet d)
(province b)
)
(:goal
(and
(craves b c)
(craves d b))
)
)


