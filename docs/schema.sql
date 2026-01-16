--
-- PostgreSQL database dump
--

\restrict dZVe8InHuQPNbNtiibegUS0bjrY0Va8grJvmpkXQQRitMZkQ6Lt3pItdR4EkwkD

-- Dumped from database version 17.4
-- Dumped by pg_dump version 17.7 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: langchain_pg_collection; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.langchain_pg_collection (
    name character varying,
    cmetadata json,
    uuid uuid NOT NULL
);


--
-- Name: langchain_pg_embedding; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.langchain_pg_embedding (
    collection_id uuid,
    embedding public.vector,
    document character varying,
    cmetadata json,
    custom_id character varying,
    uuid uuid NOT NULL
);


--
-- Name: langchain_pg_collection langchain_pg_collection_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.langchain_pg_collection
    ADD CONSTRAINT langchain_pg_collection_pkey PRIMARY KEY (uuid);


--
-- Name: langchain_pg_embedding langchain_pg_embedding_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.langchain_pg_embedding
    ADD CONSTRAINT langchain_pg_embedding_pkey PRIMARY KEY (uuid);


--
-- Name: langchain_pg_embedding langchain_pg_embedding_collection_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.langchain_pg_embedding
    ADD CONSTRAINT langchain_pg_embedding_collection_id_fkey FOREIGN KEY (collection_id) REFERENCES public.langchain_pg_collection(uuid) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

\unrestrict dZVe8InHuQPNbNtiibegUS0bjrY0Va8grJvmpkXQQRitMZkQ6Lt3pItdR4EkwkD

