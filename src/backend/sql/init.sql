-- ============================================================
-- Lideranças Empáticas — schema base
-- Ordem: extensions → enums → tables → indexes → triggers → RLS off
-- Idempotente: pode ser reaplicado sem destruir dados existentes.
-- ============================================================

-- 1. Extensões -----------------------------------------------
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 2. ENUMs ---------------------------------------------------
DO $$ BEGIN
    CREATE TYPE food_category AS ENUM
        ('arroz', 'feijao', 'acucar', 'macarrao', 'oleo', 'fuba');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE user_role AS ENUM ('professor', 'aluno');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE period_type AS ENUM ('matutino', 'noturno');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 3. users ---------------------------------------------------
CREATE TABLE IF NOT EXISTS public.users (
    id            UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    email         TEXT         NOT NULL UNIQUE,
    role          user_role    NOT NULL,
    ra            TEXT         NOT NULL UNIQUE,
    full_name     TEXT         NOT NULL,
    course        TEXT         CHECK (course IN ('Administração','Ciências Contábeis','Ciências Econômicas')),
    semester      SMALLINT     CHECK (semester BETWEEN 1 AND 8),
    period        period_type  NOT NULL,
    password_hash TEXT         NOT NULL,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT users_role_email_ra_check CHECK (
        CASE role
            WHEN 'professor' THEN email LIKE '%@fecap.br'      AND ra ~ '^\d{6}$'
            WHEN 'aluno'     THEN email LIKE '%@edu.fecap.br'  AND ra ~ '^\d{8}$'
        END
    )
);

-- 4. groups --------------------------------------------------
CREATE TABLE IF NOT EXISTS public.groups (
    id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    name       TEXT        NOT NULL UNIQUE,
    created_by UUID        NOT NULL REFERENCES public.users(id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 5. group_members (PK composta) -----------------------------
CREATE TABLE IF NOT EXISTS public.group_members (
    group_id  UUID        NOT NULL REFERENCES public.groups(id) ON DELETE CASCADE,
    user_id   UUID        NOT NULL REFERENCES public.users(id)  ON DELETE CASCADE,
    joined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (group_id, user_id)
);

-- 6. detection_sessions --------------------------------------
CREATE TABLE IF NOT EXISTS public.detection_sessions (
    id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    group_id   UUID        NOT NULL REFERENCES public.groups(id) ON DELETE CASCADE,
    started_by UUID        NOT NULL REFERENCES public.users(id)  ON DELETE RESTRICT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at   TIMESTAMPTZ
);

-- 7. evidences -----------------------------------------------
CREATE TABLE IF NOT EXISTS public.evidences (
    id          UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID          NOT NULL REFERENCES public.detection_sessions(id) ON DELETE CASCADE,
    group_id    UUID          NOT NULL REFERENCES public.groups(id)             ON DELETE CASCADE,
    category    food_category NOT NULL,
    frame_url   TEXT          NOT NULL,
    confidence  NUMERIC(4,3)  NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    detected_at TIMESTAMPTZ   NOT NULL,
    dedup_hash  TEXT          NOT NULL UNIQUE,
    created_at  TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- 8. refresh_tokens ------------------------------------------
CREATE TABLE IF NOT EXISTS public.refresh_tokens (
    id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    UUID        NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    token_hash TEXT        NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked    BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 9. Indexes -------------------------------------------------
CREATE INDEX IF NOT EXISTS evidences_group_detected_idx
    ON public.evidences (group_id, detected_at DESC);
CREATE INDEX IF NOT EXISTS evidences_category_idx
    ON public.evidences (category);
CREATE INDEX IF NOT EXISTS evidences_session_idx
    ON public.evidences (session_id);
CREATE INDEX IF NOT EXISTS users_email_idx
    ON public.users (email);
CREATE INDEX IF NOT EXISTS refresh_tokens_user_active_idx
    ON public.refresh_tokens (user_id, revoked);

-- 10. Trigger updated_at em users ----------------------------
CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS users_set_updated_at ON public.users;
CREATE TRIGGER users_set_updated_at
    BEFORE UPDATE ON public.users
    FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- 11. RLS desabilitado em todas (auth é do FastAPI) ----------
ALTER TABLE public.users              DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.groups             DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.group_members      DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.detection_sessions DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.evidences          DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.refresh_tokens     DISABLE ROW LEVEL SECURITY;
