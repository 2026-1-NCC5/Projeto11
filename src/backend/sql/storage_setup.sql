-- ============================================================
-- Storage: bucket "frames" (privado) + policies
-- Idempotente.
-- ============================================================

-- 1. Cleanup do bucket legado "evidencias"
--    O Supabase BLOQUEIA delete direto via SQL em storage.buckets/objects
--    (storage.protect_delete()). Apague pelo Supabase Studio:
--    Storage → evidencias → Empty bucket → Delete bucket.

-- 2. Cria bucket "frames" privado
INSERT INTO storage.buckets (id, name, public)
VALUES ('frames', 'frames', false)
ON CONFLICT (id) DO UPDATE SET public = EXCLUDED.public;

-- 3. Policies em storage.objects
-- 3a. Leitura permitida apenas a usuários autenticados via Supabase
--     (na prática o frontend lê via FastAPI, mas mantemos a policy para
--     facilitar inspeção via Studio).
DROP POLICY IF EXISTS "frames_authenticated_read" ON storage.objects;
CREATE POLICY "frames_authenticated_read"
    ON storage.objects
    FOR SELECT
    TO authenticated
    USING (bucket_id = 'frames');

-- 3b. Escrita (INSERT/UPDATE/DELETE) NÃO tem policy.
--     Como RLS está habilitado em storage.objects por padrão e não há policy
--     permissiva, anon e authenticated NÃO conseguem escrever.
--     O service_role (usado pelo cv_detector) bypassa RLS automaticamente,
--     então a escrita do detector continua funcionando.
