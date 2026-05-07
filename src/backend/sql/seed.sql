-- ============================================================
-- Seed de desenvolvimento — Lideranças Empáticas
--
-- Conteúdo:
--   * 2 professores  (RA 6 dígitos, @fecap.br)
--   * 12 alunos      (RA AAMMXXXX, @edu.fecap.br) — 3 cursos × 2 períodos
--   * 3 grupos       (4 membros cada)
--   * 2 detection_sessions  (1 ativa, 1 encerrada)
--   * 30 evidences   (15 + 15) cobrindo as 6 categorias
--
-- Senha de TODOS os usuários: "Senha@123"
--   (hash bcrypt gerado no momento via crypt() + gen_salt('bf') do pgcrypto)
--
-- IDs UUID fixos para facilitar inspeção e referência cruzada.
-- ============================================================

-- Limpeza idempotente (RESTART IDENTITY não importa porque os PKs são UUID).
TRUNCATE TABLE
    public.evidences,
    public.refresh_tokens,
    public.detection_sessions,
    public.group_members,
    public.groups,
    public.users
RESTART IDENTITY CASCADE;

-- ----------- USERS: PROFESSORES -----------
INSERT INTO public.users (id, email, role, ra, full_name, course, semester, period, password_hash) VALUES
    ('11111111-1111-1111-1111-111111111001', 'marcos.nakatsugawa@fecap.br', 'professor', '100001', 'Marcos Minoru Nakatsugawa', NULL, NULL, 'matutino', crypt('Senha@123', gen_salt('bf'))),
    ('11111111-1111-1111-1111-111111111002', 'rafael.rossetti@fecap.br',    'professor', '100002', 'Rafael Diogo Rossetti',     NULL, NULL, 'noturno',  crypt('Senha@123', gen_salt('bf')));

-- ----------- USERS: ALUNOS -----------
-- Grupo Alfa: Administração, matutino, 5º semestre
INSERT INTO public.users (id, email, role, ra, full_name, course, semester, period, password_hash) VALUES
    ('22222222-2222-2222-2222-222222222001', 'flavia.costa@edu.fecap.br',     'aluno', '24021001', 'Flávia Costa',          'Administração', 5, 'matutino', crypt('Senha@123', gen_salt('bf'))),
    ('22222222-2222-2222-2222-222222222002', 'guilherme.muniz@edu.fecap.br',  'aluno', '24021002', 'Guilherme Muniz',       'Administração', 5, 'matutino', crypt('Senha@123', gen_salt('bf'))),
    ('22222222-2222-2222-2222-222222222003', 'lucas.moreira@edu.fecap.br',    'aluno', '24021003', 'Lucas Moreira',         'Administração', 5, 'matutino', crypt('Senha@123', gen_salt('bf'))),
    ('22222222-2222-2222-2222-222222222004', 'maria.foloni@edu.fecap.br',     'aluno', '24021004', 'Maria Eduarda Foloni',  'Administração', 5, 'matutino', crypt('Senha@123', gen_salt('bf')));

-- Grupo Beta: Ciências Contábeis, noturno, 6º semestre
INSERT INTO public.users (id, email, role, ra, full_name, course, semester, period, password_hash) VALUES
    ('22222222-2222-2222-2222-222222222005', 'ana.silva@edu.fecap.br',        'aluno', '23042001', 'Ana Silva',             'Ciências Contábeis',  6, 'noturno',  crypt('Senha@123', gen_salt('bf'))),
    ('22222222-2222-2222-2222-222222222006', 'bruno.alves@edu.fecap.br',      'aluno', '23042002', 'Bruno Alves',           'Ciências Contábeis',  6, 'noturno',  crypt('Senha@123', gen_salt('bf'))),
    ('22222222-2222-2222-2222-222222222007', 'carla.dias@edu.fecap.br',       'aluno', '23042003', 'Carla Dias',            'Ciências Contábeis',  6, 'noturno',  crypt('Senha@123', gen_salt('bf'))),
    ('22222222-2222-2222-2222-222222222008', 'diego.farias@edu.fecap.br',     'aluno', '23042004', 'Diego Farias',          'Ciências Contábeis',  6, 'noturno',  crypt('Senha@123', gen_salt('bf')));

-- Grupo Gama: Ciências Econômicas, matutino, 4º semestre
INSERT INTO public.users (id, email, role, ra, full_name, course, semester, period, password_hash) VALUES
    ('22222222-2222-2222-2222-222222222009', 'eduardo.gomes@edu.fecap.br',    'aluno', '24023001', 'Eduardo Gomes',         'Ciências Econômicas', 4, 'matutino', crypt('Senha@123', gen_salt('bf'))),
    ('22222222-2222-2222-2222-222222222010', 'fernanda.hayashi@edu.fecap.br', 'aluno', '24023002', 'Fernanda Hayashi',      'Ciências Econômicas', 4, 'matutino', crypt('Senha@123', gen_salt('bf'))),
    ('22222222-2222-2222-2222-222222222011', 'gustavo.iaranda@edu.fecap.br',  'aluno', '24023003', 'Gustavo Iaranda',       'Ciências Econômicas', 4, 'matutino', crypt('Senha@123', gen_salt('bf'))),
    ('22222222-2222-2222-2222-222222222012', 'helena.jorge@edu.fecap.br',     'aluno', '24023004', 'Helena Jorge',          'Ciências Econômicas', 4, 'matutino', crypt('Senha@123', gen_salt('bf')));

-- ----------- GROUPS -----------
INSERT INTO public.groups (id, name, created_by) VALUES
    ('33333333-3333-3333-3333-333333333001', 'Equipe Alfa',  '22222222-2222-2222-2222-222222222001'),
    ('33333333-3333-3333-3333-333333333002', 'Equipe Beta',  '22222222-2222-2222-2222-222222222005'),
    ('33333333-3333-3333-3333-333333333003', 'Equipe Gama',  '22222222-2222-2222-2222-222222222009');

-- ----------- GROUP MEMBERS (4 por grupo) -----------
INSERT INTO public.group_members (group_id, user_id) VALUES
    ('33333333-3333-3333-3333-333333333001', '22222222-2222-2222-2222-222222222001'),
    ('33333333-3333-3333-3333-333333333001', '22222222-2222-2222-2222-222222222002'),
    ('33333333-3333-3333-3333-333333333001', '22222222-2222-2222-2222-222222222003'),
    ('33333333-3333-3333-3333-333333333001', '22222222-2222-2222-2222-222222222004'),
    ('33333333-3333-3333-3333-333333333002', '22222222-2222-2222-2222-222222222005'),
    ('33333333-3333-3333-3333-333333333002', '22222222-2222-2222-2222-222222222006'),
    ('33333333-3333-3333-3333-333333333002', '22222222-2222-2222-2222-222222222007'),
    ('33333333-3333-3333-3333-333333333002', '22222222-2222-2222-2222-222222222008'),
    ('33333333-3333-3333-3333-333333333003', '22222222-2222-2222-2222-222222222009'),
    ('33333333-3333-3333-3333-333333333003', '22222222-2222-2222-2222-222222222010'),
    ('33333333-3333-3333-3333-333333333003', '22222222-2222-2222-2222-222222222011'),
    ('33333333-3333-3333-3333-333333333003', '22222222-2222-2222-2222-222222222012');

-- ----------- DETECTION SESSIONS -----------
-- Session 1: Equipe Alfa — encerrada (3h atrás → 2h atrás)
-- Session 2: Equipe Beta — ativa    (30 min atrás, ended_at NULL)
-- (Equipe Gama não tem sessão neste seed; está propositalmente vazia.)
INSERT INTO public.detection_sessions (id, group_id, started_by, started_at, ended_at) VALUES
    ('44444444-4444-4444-4444-444444444001', '33333333-3333-3333-3333-333333333001',
        '22222222-2222-2222-2222-222222222001',
        NOW() - INTERVAL '3 hours', NOW() - INTERVAL '2 hours'),
    ('44444444-4444-4444-4444-444444444002', '33333333-3333-3333-3333-333333333002',
        '22222222-2222-2222-2222-222222222005',
        NOW() - INTERVAL '30 minutes', NULL);

-- ----------- EVIDENCES (30 = 15 + 15) -----------
-- Distribuição:
--   Session 1 (Alfa, encerrada): 15 evidências cobrindo todas as 6 categorias
--   Session 2 (Beta, ativa):     15 evidências cobrindo todas as 6 categorias
INSERT INTO public.evidences (session_id, group_id, category, frame_url, confidence, detected_at, dedup_hash) VALUES
    -- Session 1 / Equipe Alfa
    ('44444444-4444-4444-4444-444444444001','33333333-3333-3333-3333-333333333001','arroz',    'groups/alfa/sess1/01.jpg', 0.913, NOW() - INTERVAL '178 minutes', 'seed_alfa_arroz_001'),
    ('44444444-4444-4444-4444-444444444001','33333333-3333-3333-3333-333333333001','arroz',    'groups/alfa/sess1/02.jpg', 0.881, NOW() - INTERVAL '174 minutes', 'seed_alfa_arroz_002'),
    ('44444444-4444-4444-4444-444444444001','33333333-3333-3333-3333-333333333001','arroz',    'groups/alfa/sess1/03.jpg', 0.927, NOW() - INTERVAL '170 minutes', 'seed_alfa_arroz_003'),
    ('44444444-4444-4444-4444-444444444001','33333333-3333-3333-3333-333333333001','feijao',   'groups/alfa/sess1/04.jpg', 0.864, NOW() - INTERVAL '165 minutes', 'seed_alfa_feijao_001'),
    ('44444444-4444-4444-4444-444444444001','33333333-3333-3333-3333-333333333001','feijao',   'groups/alfa/sess1/05.jpg', 0.892, NOW() - INTERVAL '160 minutes', 'seed_alfa_feijao_002'),
    ('44444444-4444-4444-4444-444444444001','33333333-3333-3333-3333-333333333001','feijao',   'groups/alfa/sess1/06.jpg', 0.901, NOW() - INTERVAL '156 minutes', 'seed_alfa_feijao_003'),
    ('44444444-4444-4444-4444-444444444001','33333333-3333-3333-3333-333333333001','acucar',   'groups/alfa/sess1/07.jpg', 0.842, NOW() - INTERVAL '150 minutes', 'seed_alfa_acucar_001'),
    ('44444444-4444-4444-4444-444444444001','33333333-3333-3333-3333-333333333001','acucar',   'groups/alfa/sess1/08.jpg', 0.870, NOW() - INTERVAL '145 minutes', 'seed_alfa_acucar_002'),
    ('44444444-4444-4444-4444-444444444001','33333333-3333-3333-3333-333333333001','macarrao', 'groups/alfa/sess1/09.jpg', 0.918, NOW() - INTERVAL '140 minutes', 'seed_alfa_macarrao_001'),
    ('44444444-4444-4444-4444-444444444001','33333333-3333-3333-3333-333333333001','macarrao', 'groups/alfa/sess1/10.jpg', 0.886, NOW() - INTERVAL '135 minutes', 'seed_alfa_macarrao_002'),
    ('44444444-4444-4444-4444-444444444001','33333333-3333-3333-3333-333333333001','macarrao', 'groups/alfa/sess1/11.jpg', 0.904, NOW() - INTERVAL '130 minutes', 'seed_alfa_macarrao_003'),
    ('44444444-4444-4444-4444-444444444001','33333333-3333-3333-3333-333333333001','oleo',     'groups/alfa/sess1/12.jpg', 0.823, NOW() - INTERVAL '125 minutes', 'seed_alfa_oleo_001'),
    ('44444444-4444-4444-4444-444444444001','33333333-3333-3333-3333-333333333001','oleo',     'groups/alfa/sess1/13.jpg', 0.857, NOW() - INTERVAL '120 minutes', 'seed_alfa_oleo_002'),
    ('44444444-4444-4444-4444-444444444001','33333333-3333-3333-3333-333333333001','fuba',     'groups/alfa/sess1/14.jpg', 0.795, NOW() - INTERVAL '115 minutes', 'seed_alfa_fuba_001'),
    ('44444444-4444-4444-4444-444444444001','33333333-3333-3333-3333-333333333001','fuba',     'groups/alfa/sess1/15.jpg', 0.812, NOW() - INTERVAL '110 minutes', 'seed_alfa_fuba_002'),

    -- Session 2 / Equipe Beta (ativa)
    ('44444444-4444-4444-4444-444444444002','33333333-3333-3333-3333-333333333002','arroz',    'groups/beta/sess2/01.jpg', 0.902, NOW() - INTERVAL '28 minutes', 'seed_beta_arroz_001'),
    ('44444444-4444-4444-4444-444444444002','33333333-3333-3333-3333-333333333002','arroz',    'groups/beta/sess2/02.jpg', 0.876, NOW() - INTERVAL '26 minutes', 'seed_beta_arroz_002'),
    ('44444444-4444-4444-4444-444444444002','33333333-3333-3333-3333-333333333002','feijao',   'groups/beta/sess2/03.jpg', 0.889, NOW() - INTERVAL '24 minutes', 'seed_beta_feijao_001'),
    ('44444444-4444-4444-4444-444444444002','33333333-3333-3333-3333-333333333002','feijao',   'groups/beta/sess2/04.jpg', 0.913, NOW() - INTERVAL '22 minutes', 'seed_beta_feijao_002'),
    ('44444444-4444-4444-4444-444444444002','33333333-3333-3333-3333-333333333002','acucar',   'groups/beta/sess2/05.jpg', 0.851, NOW() - INTERVAL '20 minutes', 'seed_beta_acucar_001'),
    ('44444444-4444-4444-4444-444444444002','33333333-3333-3333-3333-333333333002','acucar',   'groups/beta/sess2/06.jpg', 0.879, NOW() - INTERVAL '18 minutes', 'seed_beta_acucar_002'),
    ('44444444-4444-4444-4444-444444444002','33333333-3333-3333-3333-333333333002','acucar',   'groups/beta/sess2/07.jpg', 0.866, NOW() - INTERVAL '16 minutes', 'seed_beta_acucar_003'),
    ('44444444-4444-4444-4444-444444444002','33333333-3333-3333-3333-333333333002','macarrao', 'groups/beta/sess2/08.jpg', 0.921, NOW() - INTERVAL '14 minutes', 'seed_beta_macarrao_001'),
    ('44444444-4444-4444-4444-444444444002','33333333-3333-3333-3333-333333333002','macarrao', 'groups/beta/sess2/09.jpg', 0.898, NOW() - INTERVAL '12 minutes', 'seed_beta_macarrao_002'),
    ('44444444-4444-4444-4444-444444444002','33333333-3333-3333-3333-333333333002','oleo',     'groups/beta/sess2/10.jpg', 0.834, NOW() - INTERVAL '10 minutes', 'seed_beta_oleo_001'),
    ('44444444-4444-4444-4444-444444444002','33333333-3333-3333-3333-333333333002','oleo',     'groups/beta/sess2/11.jpg', 0.861, NOW() - INTERVAL '8 minutes',  'seed_beta_oleo_002'),
    ('44444444-4444-4444-4444-444444444002','33333333-3333-3333-3333-333333333002','oleo',     'groups/beta/sess2/12.jpg', 0.847, NOW() - INTERVAL '6 minutes',  'seed_beta_oleo_003'),
    ('44444444-4444-4444-4444-444444444002','33333333-3333-3333-3333-333333333002','fuba',     'groups/beta/sess2/13.jpg', 0.808, NOW() - INTERVAL '4 minutes',  'seed_beta_fuba_001'),
    ('44444444-4444-4444-4444-444444444002','33333333-3333-3333-3333-333333333002','fuba',     'groups/beta/sess2/14.jpg', 0.825, NOW() - INTERVAL '3 minutes',  'seed_beta_fuba_002'),
    ('44444444-4444-4444-4444-444444444002','33333333-3333-3333-3333-333333333002','fuba',     'groups/beta/sess2/15.jpg', 0.839, NOW() - INTERVAL '2 minutes',  'seed_beta_fuba_003');
