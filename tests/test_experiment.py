"""Recorded scopes and safeguards; these checks never load a model."""
from types import SimpleNamespace
import json
import pytest
from clabsa.experiment import jobs_for, freeze_manifest, scale_review_loss, protocol, job_lock
from clabsa.experiment import prediction_prefix,commit_prefix_repair,weighted_trainer_class


def options(**values):
    return SimpleNamespace(**(dict(model='qwen',domains=None,tasks=None,seeds=None,
        empty_weight=1.,cross_domain=False,stage='plan',checkpoint_every=100)|values))


def test_repeated_scopes_are_distinct():
    qwen=jobs_for(options())
    assert len(qwen)==18 and len({j.name for j in qwen})==18
    assert len(jobs_for(options(model='mt5')))==18
    assert len(jobs_for(options(model='gemma')))==3
    assert len(jobs_for(options(empty_weight=.25)))==6
    assert len(jobs_for(options(cross_domain=True)))==6
    assert protocol(qwen[0])['scheduler']=='linear'


def test_transfer_cannot_retrain_or_expand_weighting():
    with pytest.raises(ValueError):jobs_for(options(cross_domain=True,stage='train'))
    with pytest.raises(ValueError):jobs_for(options(empty_weight=.25,domains=['laptop']))
    with pytest.raises(ValueError):jobs_for(options(model='gemma',domains=['food']))


def test_existing_identity_cannot_be_silently_changed(tmp_path):
    path=tmp_path/'identity.json'
    freeze_manifest(path,{'seed':13})
    freeze_manifest(path,{'seed':13})
    with pytest.raises(ValueError):freeze_manifest(path,{'seed':42})
    assert json.loads(path.read_text())=={'seed':13}


def test_review_weight_scales_actual_loss_and_gradient():
    torch=pytest.importorskip('torch')
    x=torch.tensor(2.,requires_grad=True)
    loss=scale_review_loss(x*x,torch.tensor([.25]))
    loss.backward()
    assert loss.item()==1 and x.grad.item()==1
    with pytest.raises(ValueError):scale_review_loss(loss,torch.tensor([.25,1.]))


def test_second_worker_cannot_enter_same_job(tmp_path):
    with job_lock(tmp_path):
        with pytest.raises(RuntimeError):
            with job_lock(tmp_path):pass
    assert not (tmp_path/'worker.lock').exists()


def test_torn_prediction_tail_is_backed_up_without_losing_valid_rows(tmp_path):
    path=tmp_path/'en.jsonl';prefix=b'{"source_index": 0}\n'
    path.write_bytes(prefix+b'{"source_ind')
    rows,length=prediction_prefix(path)
    assert rows==[{'source_index':0}] and length==len(prefix)
    commit_prefix_repair(path,length)
    assert path.read_bytes()==prefix
    assert next(tmp_path.glob('*.incomplete-*')).read_bytes()==b'{"source_ind'
    path.write_bytes(b'broken\n'+prefix)
    with pytest.raises(ValueError):prediction_prefix(path)


def test_trainer_override_applies_weight_and_preserves_outputs():
    torch=pytest.importorskip('torch')
    class Base:
        def compute_loss(self,model,inputs,return_outputs=False,**kwargs):
            assert 'review_weight' not in inputs
            return (model.square(),{'answer':1}) if return_outputs else model.square()
    trainer=weighted_trainer_class(Base)()
    x=torch.tensor(2.,requires_grad=True)
    loss,output=trainer.compute_loss(x,{'review_weight':torch.tensor([.25])},return_outputs=True)
    loss.backward()
    assert loss.item()==1 and x.grad.item()==1 and output=={'answer':1}
    assert trainer.total_visits==1 and trainer.weighted_visits==1


def test_recorded_training_transport_is_explicit():
    from clabsa.train_qlora import recorded_training_answer
    assert recorded_training_answer([(' Recommended','positive')],'uabsa')=='[{"entity":"Recommended","sentiment":"positive"}]'
    assert recorded_training_answer([('4,500mAh','capacity','neutral')],'tasd')=='[{"entity":"4","category":"500mAh","sentiment":"capacity"}]'
