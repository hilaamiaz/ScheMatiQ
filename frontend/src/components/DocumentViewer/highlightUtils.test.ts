import { findHighlightRange, findHighlightRanges } from './highlightUtils';

describe('findHighlightRange', () => {
  it('finds an exact case-insensitive match', () => {
    const text = 'The quick brown fox jumps over the lazy dog.';
    expect(findHighlightRange(text, 'BROWN fox')).toEqual([10, 19]);
  });

  it('finds a whitespace/quote-normalized match', () => {
    const text = 'The model achieved  86.4%  accuracy on MMLU.';
    expect(findHighlightRange(text, '"achieved 86.4% accuracy"')).not.toBeNull();
  });

  it('returns null when nothing resembles the query, even via the prefix fallback', () => {
    const text = 'This paper discusses climate change and rainfall patterns.';
    expect(findHighlightRange(text, 'GPT-4 achieves 86.4% accuracy on MMLU')).toBeNull();
  });
});

describe('findHighlightRanges', () => {
  const text = 'Intro. The model achieved an accuracy of 86.4 percent on MMLU. Outro.';

  it('uses a valid char_start/char_end offset directly instead of re-searching text', () => {
    const start = text.indexOf('accuracy of 86.4 percent on');
    const end = start + 'accuracy of 86.4 percent on'.length;
    // A paraphrase ("86.4%" vs the source's "86.4 percent") that this
    // module's own text search cannot find on its own -- confirm that first.
    expect(findHighlightRange(text, 'accuracy of 86.4% on MMLU')).toBeNull();

    const ranges = findHighlightRanges(text, [
      { text: 'accuracy of 86.4% on MMLU', char_start: start, char_end: end },
    ]);
    expect(ranges).toEqual([[start, end]]);
  });

  it('falls back to text search when char_start/char_end are absent', () => {
    const ranges = findHighlightRanges(text, [{ text: 'accuracy of 86.4 percent on' }]);
    expect(ranges).toHaveLength(1);
    expect(text.slice(...ranges[0])).toBe('accuracy of 86.4 percent on');
  });

  it('falls back to text search when the offset is out of range', () => {
    const ranges = findHighlightRanges(text, [
      { text: 'accuracy of 86.4 percent on', char_start: 9999, char_end: 10010 },
    ]);
    expect(ranges).toHaveLength(1);
    expect(text.slice(...ranges[0])).toBe('accuracy of 86.4 percent on');
  });

  it('falls back to text search when the offset points at unrelated text (stale offset guard)', () => {
    // char_start/char_end are in-range but point at "Intro." while the
    // excerpt text is about something else entirely -- the first-word sanity
    // check should reject this offset and fall back to a real search.
    const ranges = findHighlightRanges(text, [
      { text: 'accuracy of 86.4 percent on', char_start: 0, char_end: 6 },
    ]);
    expect(ranges).toHaveLength(1);
    expect(text.slice(...ranges[0])).toBe('accuracy of 86.4 percent on');
  });

  it('drops overlapping ranges, keeping the first', () => {
    const ranges = findHighlightRanges(text, [{ text: 'accuracy of' }, { text: 'accuracy of 86.4 percent on' }]);
    expect(ranges).toHaveLength(1);
  });

  it('returns an empty array when text or excerpts are missing', () => {
    expect(findHighlightRanges(null, [{ text: 'x' }])).toEqual([]);
    expect(findHighlightRanges(text, null)).toEqual([]);
    expect(findHighlightRanges(text, [])).toEqual([]);
  });
});
