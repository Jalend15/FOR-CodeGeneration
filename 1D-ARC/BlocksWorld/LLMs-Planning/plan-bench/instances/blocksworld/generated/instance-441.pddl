(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects e l h c b g d i)
(:init 
(handempty)
(ontable e)
(ontable l)
(ontable h)
(ontable c)
(ontable b)
(ontable g)
(ontable d)
(ontable i)
(clear e)
(clear l)
(clear h)
(clear c)
(clear b)
(clear g)
(clear d)
(clear i)
)
(:goal
(and
(on e l)
(on l h)
(on h c)
(on c b)
(on b g)
(on g d)
(on d i)
)))