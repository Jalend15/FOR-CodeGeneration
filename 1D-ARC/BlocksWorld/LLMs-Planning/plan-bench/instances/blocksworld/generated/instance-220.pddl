(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects e k j g a c h b l f d i)
(:init 
(handempty)
(ontable e)
(ontable k)
(ontable j)
(ontable g)
(ontable a)
(ontable c)
(ontable h)
(ontable b)
(ontable l)
(ontable f)
(ontable d)
(ontable i)
(clear e)
(clear k)
(clear j)
(clear g)
(clear a)
(clear c)
(clear h)
(clear b)
(clear l)
(clear f)
(clear d)
(clear i)
)
(:goal
(and
(on e k)
(on k j)
(on j g)
(on g a)
(on a c)
(on c h)
(on h b)
(on b l)
(on l f)
(on f d)
(on d i)
)))