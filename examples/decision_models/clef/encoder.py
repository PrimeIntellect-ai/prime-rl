"""Clef's encoder (``[data.encoder]``, ``import_path = "decision_models.clef.encoder.ClefEncoder"``).

The prompt, its markup and its tokens are Clef's own: ``encode_record`` in ``joint_schema_model.py``, an unmodified
copy of the file in https://huggingface.co/Cloudflare/clef-flash (Apache-2.0; see THIRD_PARTY.md).
"""

from decision_models.clef.joint_schema_model import encode_record

from prime_rl.trainer.sft.system_one.data import EncodedRecord, SystemOneEncoder
from prime_rl.trainer.sft.system_one.record import SystemOneRecord
from prime_rl.utils.logger import get_logger


def clef_request(record: SystemOneRecord) -> dict:
    """The record as the request ``encode_record`` takes: an id, the state and the System One questions."""
    questions = {
        question.id: {"type": question.type, "instructions": question.instructions, "criteria": question.criteria}
        for question in record.questions
    }
    return {"id": str(record.id), "state": record.state, "questions": questions}


class ClefEncoder(SystemOneEncoder):
    """The whole record is one causal prompt and one attention segment: the state, then every question's schema.
    A question's read spans are its instruction followed by one span per option, in Clef's option order (noul
    ``true``, ``false``; choice ids sorted; score levels ascending). Clef shortens a long state to fit ``max_tokens``;
    a record whose schema alone does not fit is skipped with a warning."""

    def __call__(self, record: SystemOneRecord, max_tokens: int) -> EncodedRecord | None:
        try:
            encoded = encode_record(self.tokenizer, clef_request(record), max_length=max_tokens)
        except ValueError as error:
            if "schema requires" not in str(error):
                raise
            get_logger().warning(f"Skipping record {record.id!r}: {error}")
            return None
        return EncodedRecord(
            input_ids=list(encoded.input_ids),
            seq_lens=[len(encoded.input_ids)],
            reads=[[question.question_span, *question.option_spans] for question in encoded.questions],
            options=[list(question.option_ids) for question in encoded.questions],
        )
