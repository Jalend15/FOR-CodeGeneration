(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects b c h l e j d)
(:init 
(handempty)
(ontable b)
(ontable c)
(ontable h)
(ontable l)
(ontable e)
(ontable j)
(ontable d)
(clear b)
(clear c)
(clear h)
(clear l)
(clear e)
(clear j)
(clear d)
)
(:goal
(and
(on b c)
(on c h)
(on h l)
(on l e)
(on e j)
(on j d)
)))