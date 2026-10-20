/**
 * loadAPI.addDocuments sends large uploads in small batches and resends a
 * batch whose request failed transiently.
 */

// axios is ESM-only and not transformed by react-scripts' jest; mock it whole.
// jest.mock is hoisted above this declaration, so create() must read mockPost
// lazily (api.ts calls create() at import time).
const mockPost = jest.fn();
jest.mock('axios', () => ({
  __esModule: true,
  default: {
    create: () => ({ post: (...args: unknown[]) => mockPost(...args), get: jest.fn() }),
  },
}));

import { loadAPI } from './api';

const file = (name: string) => new File(['x'], name);

const okFor = (form: FormData) => ({
  data: {
    status: 'success',
    message: '',
    uploaded_files: (form.getAll('files') as File[]).map(f => f.name.replace(/\.pdf$/, '.txt')),
    warnings: [],
    documents_directory: '/docs',
  },
});

beforeEach(() => {
  mockPost.mockReset();
  jest.spyOn(global, 'setTimeout').mockImplementation(((fn: () => void) => {
    fn();
    return 0;
  }) as unknown as typeof setTimeout);
  jest.spyOn(console, 'warn').mockImplementation(() => undefined);
});

afterEach(() => {
  jest.restoreAllMocks();
});

test('splits the upload into batches of four', async () => {
  mockPost.mockImplementation((_url: string, form: FormData) => Promise.resolve(okFor(form)));
  const files = Array.from({ length: 10 }, (_, i) => file(`p${i}.pdf`));

  const result = await loadAPI.addDocuments('s1', files);

  expect(mockPost).toHaveBeenCalledTimes(3);
  expect(result.uploaded_files.sort()).toEqual(files.map(f => f.name.replace('.pdf', '.txt')).sort());
  expect(result.failed_files).toBeUndefined();
});

test('resends a batch after a network error or 5xx', async () => {
  let calls = 0;
  mockPost.mockImplementation((_url: string, form: FormData) => {
    calls++;
    if (calls === 1) return Promise.reject({ code: 'ERR_NETWORK' });
    if (calls === 2) return Promise.reject({ response: { status: 502 } });
    return Promise.resolve(okFor(form));
  });

  const result = await loadAPI.addDocuments('s1', [file('a.pdf'), file('b.pdf')]);

  expect(mockPost).toHaveBeenCalledTimes(3);
  expect(result.uploaded_files.sort()).toEqual(['a.txt', 'b.txt']);
});

test('a rejected batch does not abort the other batches', async () => {
  let calls = 0;
  mockPost.mockImplementation((_url: string, form: FormData) => {
    calls++;
    if (calls === 1) {
      return Promise.reject({ response: { status: 400, data: { detail: 'bad files' } } });
    }
    return Promise.resolve(okFor(form));
  });

  const result = await loadAPI.addDocuments('s1', Array.from({ length: 6 }, (_, i) => file(`p${i}.pdf`)));

  expect(mockPost).toHaveBeenCalledTimes(2); // a 400 is not retried
  expect(result.uploaded_files).toHaveLength(2);
  expect(result.failed_files).toHaveLength(4);
});

test('throws when every batch fails', async () => {
  mockPost.mockRejectedValue({ response: { status: 400, data: { detail: 'nope' } } });
  await expect(loadAPI.addDocuments('s1', [file('a.pdf')])).rejects.toBeTruthy();
});
